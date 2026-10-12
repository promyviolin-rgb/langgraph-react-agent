#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mały serwer HTTP (FastAPI) wystawiający agenta LangGraph dla n8n.

To jest "mózg do trudnych zadań". n8n dalej robi wszystko, co proste
(Telegram, Gmail, kalendarz, wyzwalacze). Gdy zadanie wymaga pętli,
refleksji albo trwałego kontekstu - n8n woła ten serwer jednym
węzłem HTTP Request.

Trwała pamięć:
    MemorySaver trzyma rozmowy tylko w RAM - restart = utrata pamięci.
    Tutaj używamy SqliteSaver: rozmowy leżą w pliku pamiec.sqlite
    i przeżywają restart serwera.

Bezpieczeństwo:
    Każde zapytanie musi mieć nagłówek X-API-Token.
    Brak tokenu w konfiguracji = serwer nie wstaje (fail closed).

Uruchomienie:
    ./run-serwer.sh
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from agent_react_langgraph import MODEL_NAME, TOOLS, _make_fake_llm, build_graph

KATALOG = Path(__file__).resolve().parent
SCIEZKA_BAZY = KATALOG / "pamiec.sqlite"


# --------------------------------------------------------------------------- #
# Konfiguracja z pliku .env (jeśli istnieje)
# --------------------------------------------------------------------------- #


def wczytaj_env() -> None:
    """Wczytuje prosty plik .env obok skryptu (nie nadpisuje środowiska)."""
    plik = KATALOG / ".env"
    if not plik.exists():
        return
    for linia in plik.read_text(encoding="utf-8").splitlines():
        linia = linia.strip()
        if not linia or linia.startswith("#") or "=" not in linia:
            continue
        klucz, wartosc = linia.split("=", 1)
        os.environ.setdefault(klucz.strip(), wartosc.strip().strip("'\""))


wczytaj_env()

TOKEN = os.getenv("LANGGRAPH_TOKEN", "")
BLOKADA = threading.Lock()  # SQLite lubi jeden zapis naraz


# --------------------------------------------------------------------------- #
# Start / stop
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def cykl_zycia(app: FastAPI):
    tryb_testowy = os.getenv("LANGGRAPH_FAKE_LLM") == "1"

    if not tryb_testowy and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "Brak OPENROUTER_API_KEY. Ustaw go w .env albo w środowisku."
        )
    if not TOKEN:
        raise RuntimeError(
            "Brak LANGGRAPH_TOKEN. Serwer nie wstaje bez tokenu (fail closed).\n"
            "Wygeneruj:  ./run-serwer.sh --nowy-token"
        )

    polaczenie = sqlite3.connect(SCIEZKA_BAZY, check_same_thread=False)
    app.state.checkpointer = SqliteSaver(polaczenie)
    # Bez tego tabele 'checkpoints'/'writes' nie istnieją i zapis jest cichy
    # (brak błędu, ale pamięć nie trafia na dysk).
    app.state.checkpointer.setup()

    # Tryb WYŁĄCZNIE testowy: atrapa modelu, zero wywołań API.
    # Służy do sprawdzenia serwera i trwałości pamięci bez ważnego klucza.
    if tryb_testowy:
        app.state.graf = build_graph(
            llm=_make_fake_llm(), checkpointer=app.state.checkpointer
        )
        tryb = "ATRAPA MODELU (test)"
    else:
        app.state.graf = build_graph(checkpointer=app.state.checkpointer)
        tryb = "produkcyjny"

    print("=" * 70)
    print("SERWER LANGGRAPH WYSTARTOWAŁ")
    print(f"  Tryb:      {tryb}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Narzędzia: {[t.name for t in TOOLS]}")
    print(f"  Pamięć:    {SCIEZKA_BAZY} (trwała, SQLite)")
    print(f"  Token:     ustawiony ({len(TOKEN)} znaków)")
    print("=" * 70)
    try:
        yield
    finally:
        polaczenie.close()


app = FastAPI(title="LangGraph dla n8n", lifespan=cykl_zycia)


# --------------------------------------------------------------------------- #
# Modele zapytań i odpowiedzi
# --------------------------------------------------------------------------- #


class Zapytanie(BaseModel):
    thread_id: str = Field(
        ..., min_length=1, max_length=200, description="Numer rozmowy (pamięć)."
    )
    message: str = Field(..., min_length=1, max_length=8000, description="Treść.")


class Odpowiedz(BaseModel):
    odpowiedz: str
    thread_id: str
    uzyte_narzedzia: list[str]
    liczba_wiadomosci: int


# --------------------------------------------------------------------------- #
# Endpointy
# --------------------------------------------------------------------------- #


def sprawdz_token(x_api_token: str | None) -> None:
    """Porównanie odporne na atak czasowy. Brak/zły token = 401."""
    if not x_api_token or not secrets.compare_digest(x_api_token, TOKEN):
        raise HTTPException(status_code=401, detail="Zły lub brakujący token.")


@app.get("/zdrowie")
def zdrowie() -> dict[str, Any]:
    """Bez tokenu - tylko sprawdzenie, czy serwer żyje."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/zapytanie", response_model=Odpowiedz)
def zapytanie(
    dane: Zapytanie,
    x_api_token: str | None = Header(default=None),
) -> Odpowiedz:
    """Główny endpoint. n8n woła go węzłem HTTP Request."""
    sprawdz_token(x_api_token)

    konfiguracja = {"configurable": {"thread_id": dane.thread_id}}
    wejscie = {"messages": [{"role": "user", "content": dane.message}]}

    # Endpoint jest sync (def), więc FastAPI sam wrzuca go do wątku roboczego.
    with BLOKADA:
        wynik = app.state.graf.invoke(wejscie, config=konfiguracja)

    wiadomosci = wynik["messages"]
    uzyte = [
        w.name
        for w in wiadomosci
        if type(w).__name__ == "ToolMessage" and getattr(w, "name", None)
    ]

    return Odpowiedz(
        odpowiedz=str(wiadomosci[-1].content),
        thread_id=dane.thread_id,
        uzyte_narzedzia=uzyte,
        liczba_wiadomosci=len(wiadomosci),
    )


@app.delete("/watek/{thread_id}")
def usun_watek(
    thread_id: str, x_api_token: str | None = Header(default=None)
) -> dict[str, str]:
    """Czyści pamięć jednej rozmowy."""
    sprawdz_token(x_api_token)
    with BLOKADA:
        app.state.checkpointer.delete_thread(thread_id)
    return {"status": "usunieto", "thread_id": thread_id}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def nowy_token() -> int:
    """Tworzy/zapisuje token do .env (prawa 600) i wypisuje go."""
    sciezka = KATALOG / ".env"
    nowy = secrets.token_urlsafe(32)

    linie: list[str] = []
    if sciezka.exists():
        linie = [
            l
            for l in sciezka.read_text(encoding="utf-8").splitlines()
            if not l.strip().startswith("LANGGRAPH_TOKEN=")
        ]
    linie.append(f"LANGGRAPH_TOKEN={nowy}")
    sciezka.write_text("\n".join(linie) + "\n", encoding="utf-8")
    sciezka.chmod(0o600)

    print(f"Zapisano token do: {sciezka} (prawa 600)")
    print(f"LANGGRAPH_TOKEN={nowy}")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--nowy-token":
        raise SystemExit(nowy_token())

    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("LANGGRAPH_HOST", "127.0.0.1"),
        port=int(os.getenv("LANGGRAPH_PORT", "8100")),
    )
