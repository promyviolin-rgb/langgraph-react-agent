#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent LangGraph z pętlą ReAct, wywoływaniem narzędzi i pamięcią wątków.

Architektura:
    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

LLM: ChatOpenAI podpięty pod API OpenRouter (klucz z OPENROUTER_API_KEY).
Pamięć: MemorySaver (checkpointer) + ten sam thread_id w kolejnych turach.

Uruchomienie:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python agent_react_langgraph.py

Tryb offline (bez klucza i bez ruchu sieciowego) - walidacja grafu:
    python agent_react_langgraph.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --------------------------------------------------------------------------- #
# Konfiguracja
# --------------------------------------------------------------------------- #

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model musi obsługiwać function/tool calling.
# Zmień bez edycji kodu:  export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = (
    "Jesteś asystentem, który ma dostęp do narzędzi: poczty Gmail, kalendarza "
    "i wyszukiwarki internetowej. Zanim odpowiesz na pytanie o maile, wydarzenia "
    "lub informacje z sieci, NAJPIERW wywołaj odpowiednie narzędzie i oprzyj "
    "odpowiedź na jego wyniku - nigdy nie zgaduj danych, których nie widziałeś. "
    "Jeśli narzędzie zwróci błąd, powiedz o tym wprost zamiast wymyślać wynik. "
    "Odpowiadaj po polsku, zwięźle i konkretnie. "
    "Wykorzystuj kontekst całej rozmowy - jeśli użytkownik pyta uzupełniająco, "
    "odwołuj się do informacji zdobytych wcześniej."
)


# --------------------------------------------------------------------------- #
# Narzędzie: pobierz_pogode
# --------------------------------------------------------------------------- #

# Kody pogodowe WMO -> opis po polsku
_WMO_CODES: dict[int, str] = {
    0: "bezchmurnie",
    1: "głównie bezchmurnie",
    2: "częściowo pochmurno",
    3: "pochmurno",
    45: "mgła",
    48: "mgła osadzająca szadź",
    51: "mżawka słaba",
    53: "mżawka umiarkowana",
    55: "mżawka intensywna",
    56: "mżawka marznąca (słaba)",
    57: "mżawka marznąca (intensywna)",
    61: "deszcz słaby",
    63: "deszcz umiarkowany",
    65: "deszcz intensywny",
    66: "deszcz marznący (słaby)",
    67: "deszcz marzący (intensywny)",
    71: "śnieg słaby",
    73: "śnieg umiarkowany",
    75: "śnieg intensywny",
    77: "krupa śnieżna",
    80: "przelotny deszcz (słaby)",
    81: "przelotny deszcz (umiarkowany)",
    82: "przelotne opady gwałtowne",
    85: "przelotny śnieg (słaby)",
    86: "przelotny śnieg (intensywny)",
    95: "burza",
    96: "burza z gradem (słaby)",
    99: "burza z gradem (silny)",
}


@tool
def pobierz_pogode(miasto: str) -> str:
    """Pobiera aktualną pogodę dla podanego miasta.

    Użyj tego narzędzia zawsze, gdy użytkownik pyta o pogodę, temperaturę,
    wiatr, opady lub warunki atmosferyczne w jakimkolwiek mieście.

    Args:
        miasto: Nazwa miasta, np. "Berlin", "Warszawa", "Kraków".

    Returns:
        Tekstowy opis aktualnej pogody (temperatura, odczuwalna, wiatr, opady)
        wraz z surowymi danymi w formacie JSON.
    """
    try:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            geo = client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": miasto, "count": 1, "language": "pl", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return f"Nie znalazłem miasta o nazwie '{miasto}'."

            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]
            label = f"{loc['name']}, {loc.get('country', '?')}"

            meteo = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "precipitation,wind_speed_10m,weather_code"
                    ),
                    "timezone": "auto",
                },
            )
            meteo.raise_for_status()
            cur = meteo.json()["current"]

        opis = _WMO_CODES.get(int(cur["weather_code"]), "nieznane warunki")
        payload: dict[str, Any] = {
            "miasto": label,
            "temperatura_C": cur["temperature_2m"],
            "odczuwalna_C": cur["apparent_temperature"],
            "opady_mm": cur["precipitation"],
            "wiatr_kmh": cur["wind_speed_10m"],
            "warunki": opis,
            "zrodlo": "open-meteo.com",
        }
        return (
            f"Pogoda w {label}: {opis}, {payload['temperatura_C']}°C "
            f"(odczuwalna {payload['odczuwalna_C']}°C), "
            f"wiatr {payload['wiatr_kmh']} km/h, opady {payload['opady_mm']} mm. "
            f"DANE_JSON={json.dumps(payload, ensure_ascii=False)}"
        )

    except Exception as exc:  # noqa: BLE001 - narzędzie nie może wywalić grafu
        return f"Nie udało się pobrać pogody dla '{miasto}'. Błąd: {exc}"


# --------------------------------------------------------------------------- #
# Narzędzia: warstwa n8n (Gmail, kalendarz, wyszukiwarka)
# --------------------------------------------------------------------------- #
#
# Te narzędzia nie integrują się z Google bezpośrednio - wołają webhooki w n8n,
# które robią to za nie. Dzięki temu logika OAuth, uprawnień i formatowania
# wiadomości żyje w jednym miejscu, a agent dostaje wąski, nazwany interfejs.
#
# To jest także granica bezpieczeństwa: model nie może wykonać niczego, czego
# nie ma na tej liście.

N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://127.0.0.1:5678").rstrip("/")
N8N_TIMEOUT = float(os.getenv("N8N_TIMEOUT", "30"))


def _n8n_post(path: str, payload: dict[str, Any]) -> str:
    """POST do webhooka n8n. Zawsze zwraca tekst, nigdy nie rzuca wyjątku.

    Narzędzie, które rzuci wyjątek, wywraca cały graf - dlatego każdy błąd
    zamieniamy na czytelny komunikat, który model może przekazać dalej.
    """
    url = f"{N8N_BASE_URL}/webhook/{path}"
    try:
        import httpx

        with httpx.Client(timeout=N8N_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            try:
                return json.dumps(response.json(), ensure_ascii=False)[:4000]
            except ValueError:
                return response.text[:4000]
    except Exception as exc:  # noqa: BLE001 - patrz docstring
        return f"Błąd wywołania n8n ({path}): {exc}"


@tool
def gmail_szukaj(q: str) -> str:
    """Szuka wiadomości w skrzynce Gmail.

    Użyj, gdy użytkownik pyta o maile, np. "znajdź maile od Ani",
    "co przyszło wczoraj", "poszukaj faktury".

    Args:
        q: Zapytanie w składni Gmaila, np. "from:ania@example.com",
           "subject:faktura", "is:unread", "nowsze_niż:7d".

    Returns:
        Lista pasujących wiadomości z identyfikatorem, tematem, nadawcą
        i fragmentem treści (JSON).
    """
    return _n8n_post("gmail-tool-szukaj-3a91c7", {"q": q})


@tool
def gmail_przeczytaj(id: str) -> str:
    """Czyta pełną treść jednej wiadomości na podstawie jej identyfikatora.

    Użyj po `gmail_szukaj`, gdy fragment nie wystarcza do odpowiedzi.
    Identyfikator pochodzi zawsze z wyniku wyszukiwania - nie wymyślaj go.

    Args:
        id: Identyfikator wiadomości zwrócony przez `gmail_szukaj`.

    Returns:
        Treść wiadomości wraz z nadawcą, tematem i datą.
    """
    return _n8n_post("gmail-tool-czytaj-5e27b4", {"id": id})


@tool
def gmail_szkic(
    akcja: Literal["utworz", "usun"],
    do: str = "",
    temat: str = "",
    tresc: str = "",
    id: str = "",
) -> str:
    """Tworzy albo usuwa szkic wiadomości w Gmailu.

    Szkic nie jest wysyłką - trafia do zakładki Wersje robocze i czeka
    na decyzję człowieka. Dlatego to bezpieczny sposób przygotowania odpowiedzi.

    Args:
        akcja: "utworz" - nowy szkic; "usun" - skasowanie istniejącego.
        do: Adres odbiorcy (dla akcji "utworz").
        temat: Temat wiadomości (dla akcji "utworz").
        tresc: Treść wiadomości (dla akcji "utworz").
        id: Identyfikator szkicu do usunięcia (dla akcji "usun").

    Returns:
        Potwierdzenie z identyfikatorem szkicu albo opis błędu.
    """
    if akcja == "utworz":
        return _n8n_post(
            "gmail-tool-szkic-9f13d8",
            {"do": do, "temat": temat, "tresc": tresc},
        )
    if akcja == "usun":
        return _n8n_post("gmail-tool-usun-szkic-4c81", {"id": id})
    return f"Nieznana akcja '{akcja}'. Dozwolone: utworz, usun."


@tool
def gmail_etykiety(
    akcja: Literal["dodaj", "oznacz"],
    id: str = "",
    etykieta_id: str = "",
    etykieta: str = "",
) -> str:
    """Nadaje etykiety wiadomościom - pojedynczo albo hurtowo.

    Args:
        akcja: "dodaj" - etykieta dla jednej wiadomości (wymaga `id`
               i `etykieta_id`); "oznacz" - etykieta dla wielu wiadomości
               (wymaga `etykieta`, opcjonalnie filtr nadawcy w `id`).
        id: Identyfikator wiadomości (akcja "dodaj") albo filtr nadawcy,
            np. "from:newsletter@x.com" (akcja "oznacz").
        etykieta_id: Identyfikator etykiety Gmaila (akcja "dodaj").
        etykieta: Nazwa etykiety, np. "Do przeczytania" (akcja "oznacz").

    Returns:
        Potwierdzenie z liczbą oznaczonych wiadomości albo opis błędu.
    """
    if akcja == "dodaj":
        return _n8n_post(
            "gmail-tool-dodaj-etykiete-2f71",
            {"id": id, "etykieta_id": etykieta_id},
        )
    if akcja == "oznacz":
        return _n8n_post("gmail-tool-oznacz-3d51", {"etykieta": etykieta, "od": id})
    return f"Nieznana akcja '{akcja}'. Dozwolone: dodaj, oznacz."


@tool
def kalendarz_dodaj(
    od: str,
    tytul: str,
    do: str = "",
    godzina_od: str = "",
    godzina_do: str = "",
    miejsce: str = "",
    opis: str = "",
) -> str:
    """Dodaje wydarzenie do kalendarza Google.

    Bez `godzina_od` wydarzenie jest całodniowe. Daty w formacie YYYY-MM-DD.

    Args:
        od: Data początkowa, np. "2026-10-20".
        tytul: Nazwa wydarzenia.
        do: Data końcowa (opcjonalna; domyślnie ten sam dzień).
        godzina_od: Godzina rozpoczęcia "HH:MM" (opcjonalna).
        godzina_do: Godzina zakończenia "HH:MM" (domyślnie +1 godzina).
        miejsce: Miejsce wydarzenia (opcjonalne).
        opis: Dodatkowy opis (opcjonalny).

    Returns:
        Potwierdzenie z identyfikatorem wydarzenia albo opis błędu.
    """
    return _n8n_post(
        "kalendarz-tool-dodaj-6f2b91",
        {
            "od": od,
            "do": do,
            "godzina_od": godzina_od,
            "godzina_do": godzina_do,
            "tytul": tytul,
            "miejsce": miejsce,
            "opis": opis,
        },
    )


@tool
def web_szukaj(zapytanie: str) -> str:
    """Szuka informacji w internecie.

    Użyj, gdy potrzebujesz danych spoza skrzynki i kalendarza: aktualności,
    godzin otwarcia, faktów, których nie ma w kontekście rozmowy.

    Args:
        zapytanie: Treść zapytania, najlepiej konkretna i w języku, w którym
            spodziewasz się wyników.

    Returns:
        Wyniki wyszukiwania (JSON) albo opis błędu.
    """
    return _n8n_post("web-tool-szukaj-7d21", {"zapytanie": zapytanie})


TOOLS = [
    gmail_szukaj,
    gmail_przeczytaj,
    gmail_szkic,
    gmail_etykiety,
    kalendarz_dodaj,
    web_szukaj,
    pobierz_pogode,  # narzędzie przykładowe: działa bez konfiguracji i bez OAuth
]


# --------------------------------------------------------------------------- #
# Stan agenta
# --------------------------------------------------------------------------- #


class AgentState(TypedDict):
    """Stan przepływający przez graf.

    `add_messages` to reducer: nowe wiadomości są doklejane do historii,
    a nie nadpisują jej - dzięki temu pamięć wątku działa poprawnie.
    """

    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------------------------------- #
# Budowa grafu
# --------------------------------------------------------------------------- #


def build_llm() -> ChatOpenAI:
    """Tworzy ChatOpenAI wskazujący na OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "BŁĄD: brak zmiennej środowiskowej OPENROUTER_API_KEY.\n"
            'Ustaw ją:  export OPENROUTER_API_KEY="sk-or-v1-..."'
        )
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=0,
        max_retries=2,
        # Nagłówki opcjonalne, ale zalecane przez OpenRouter (atrybucja appki).
        default_headers={
            "HTTP-Referer": "https://localhost/langgraph-react",
            "X-Title": "LangGraph ReAct Agent",
        },
    )


def build_graph(llm=None, checkpointer=None):
    """Buduje i kompiluje graf ReAct.

    Args:
        llm: model do użycia (domyślnie ChatOpenAI -> OpenRouter).
        checkpointer: pamięć wątków. Domyślnie MemorySaver (tylko RAM,
            ginie po restarcie). Serwer HTTP podaje tu SqliteSaver,
            żeby pamięć przeżyła restart.
    """
    if llm is None:
        llm = build_llm()
    if checkpointer is None:
        checkpointer = MemorySaver()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Węzeł agenta: LLM decyduje - odpowiedzieć czy wywołać narzędzie."""
        wiadomosci = list(state["messages"])
        # Instrukcja systemowa wstrzykiwana raz, na początek rozmowy.
        # (Bez tego SYSTEM_PROMPT byłby martwym kodem i model go nie widzi.)
        if not any(isinstance(m, SystemMessage) for m in wiadomosci):
            wiadomosci = [SystemMessage(content=SYSTEM_PROMPT), *wiadomosci]
        response = llm_with_tools.invoke(wiadomosci)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition: jeśli LLM zwrócił tool_calls -> "tools", w innym razie -> END
    builder.add_conditional_edges("agent", tools_condition)
    # Pełna pętla ReAct: po wykonaniu narzędzi wracamy do agenta.
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Test dwuetapowy (pamięć wątku)
# --------------------------------------------------------------------------- #

WIADOMOSC_1 = "Jaka jest pogoda w Berlinie?"
WIADOMOSC_2 = "A jak się tam dzisiaj ubrać?"


def _print_trace(result: dict[str, Any], od_indeksu: int) -> None:
    """Wypisuje nowe wiadomości wygenerowane w tej turze."""
    for msg in result["messages"][od_indeksu:]:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  [{kind}] -> wywołanie narzędzia: {tc['name']}({tc['args']})")
        if kind == "ToolMessage":
            tresc = str(msg.content).replace("\n", " ")
            print(f"  [{kind}:{msg.name}] {tresc[:300]}")
        elif getattr(msg, "content", ""):
            tresc = str(msg.content).replace("\n", " ")
            print(f"  [{kind}] {tresc[:600]}")


def run_thread_test(graph, thread_id: str = "berlin-thread-1") -> dict[str, Any]:
    """Dwuinstancyjny test: dwie wiadomości, ten sam thread_id."""
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 78)
    print(f"TEST PAMIĘCI WĄTKU  |  MODEL: {MODEL_NAME}  |  thread_id='{thread_id}'")
    print("=" * 78)

    # --- Tura 1: pytanie o pogodę (agent ma użyć narzędzia) ---
    print(f"\n>>> WIADOMOŚĆ 1: {WIADOMOSC_1}")
    before = len(graph.get_state(config).values.get("messages", []))
    r1 = graph.invoke(
        {"messages": [HumanMessage(content=WIADOMOSC_1)]}, config=config
    )
    _print_trace(r1, before)

    uzyto_narzedzia = any(
        getattr(m, "name", None) == "pobierz_pogode" for m in r1["messages"]
    )
    print(f"\n[WERYFIKACJA] Agent użył narzędzia pobierz_pogode: {uzyto_narzedzia}")

    # --- Tura 2: pytanie uzupełniające (agent ma użyć pamięci) ---
    print(f"\n>>> WIADOMOŚĆ 2: {WIADOMOSC_2}")
    before2 = len(graph.get_state(config).values.get("messages", []))
    r2 = graph.invoke(
        {"messages": [HumanMessage(content=WIADOMOSC_2)]}, config=config
    )
    _print_trace(r2, before2)

    # --- Weryfikacja pamięci ---
    final_messages = r2["messages"]
    liczba = len(final_messages)
    ma_historie = liczba >= 4  # human1, ai(tool_call), tool, ai  + human2, ai
    print("\n" + "-" * 78)
    print("[WERYFIKACJA PAMIĘCI]")
    print(f"  Liczba wiadomości w wątku '{thread_id}': {liczba}")
    print(f"  Pierwsza wiadomość wciąż w stanie: {final_messages[0].content!r}")
    print(f"  Historia zachowana w checkpointerze: {ma_historie}")

    return {
        "thread_id": thread_id,
        "uzyto_narzedzia": uzyto_narzedzia,
        "liczba_wiadomosci": liczba,
        "pamiec_ok": ma_historie,
        "odpowiedz_1": r1["messages"][-1].content,
        "odpowiedz_2": r2["messages"][-1].content,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _make_fake_llm():
    """Skryptowana atrapa LLM do walidacji offline.

    Tura 1: zwraca tool_call do `pobierz_pogode` (jak prawdziwy model).
    Tura 2: zwraca już finalną odpowiedź tekstową.
    Dzięki temu offline przechodzimy CAŁĄ pętlę ReAct: agent -> tools -> agent.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class _ScriptedToolCallingModel(BaseChatModel):
        responses: list[Any] = []
        _idx: int = PrivateAttr(default=0)

        @property
        def _llm_type(self) -> str:
            return "scripted-tool-calling"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            i = min(self._idx, len(self.responses) - 1)
            self._idx += 1
            return ChatResult(
                generations=[ChatGeneration(message=self.responses[i])]
            )

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    return _ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "pobierz_pogode",
                        "args": {"miasto": "Berlin"},
                        "id": "call_check_1",
                    }
                ],
            ),
            AIMessage(content="Atrapa LLM: pogoda pobrana, pełna pętla ReAct OK."),
        ]
    )


def tryb_check() -> int:
    """Walidacja grafu bez wywołania LLM (bez klucza API)."""
    print("TRYB --check: walidacja struktury grafu i pełnej pętli ReAct offline.\n")

    fake = _make_fake_llm()
    graph = build_graph(fake)

    nodes = sorted(graph.get_graph().nodes)
    edges = sorted((e.source, e.target) for e in graph.get_graph().edges)
    print(f"Narzędzia: {[t.name for t in TOOLS]}")
    print(f"Węzły: {nodes}")
    print("Krawędzie:")
    for src, dst in edges:
        print(f"  {src} -> {dst}")

    out = graph.invoke(
        {"messages": [HumanMessage(content="test")]},
        config={"configurable": {"thread_id": "check-1"}},
    )
    print(f"\nInvoke z atrapą LLM OK, wiadomości: {len(out['messages'])}")
    print("Checkpointer: MemorySaver ->", type(graph.checkpointer).__name__)

    assert "agent" in nodes and "tools" in nodes, "brak węzłów agent/tools"
    assert ("tools", "agent") in edges, "brak pętli ReAct tools -> agent"
    assert any(s == "agent" and d == "tools" for s, d in edges), "brak warunku do tools"
    # W langgraph 1.x MemorySaver to alias klasy InMemorySaver.
    cp_name = type(graph.checkpointer).__name__
    assert cp_name in {"MemorySaver", "InMemorySaver"}, f"zły checkpointer: {cp_name}"
    print(f"Checkpointer OK ({cp_name} == MemorySaver).")

    # Trwałość pamięci: druga tura na tym samym thread_id widzi historię.
    cfg = {"configurable": {"thread_id": "check-1"}}
    r2 = graph.invoke({"messages": [HumanMessage(content="drugie pytanie")]}, config=cfg)
    stan = graph.get_state(cfg).values["messages"]
    print(f"Po 2. turze w stanie wątku 'check-1': {len(stan)} wiadomości.")
    assert len(stan) > len(out["messages"]), "checkpointer nie zapisał historii"
    print("Pamięć wątku (MemorySaver) zachowana między turami.")

    print("\n✅ Wszystkie asercje struktury grafu i pamięci przeszły.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent LangGraph ReAct + pamięć wątku")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Waliduj graf bez wywołania LLM (nie wymaga OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--thread-id", default="berlin-thread-1", help="Identyfikator wątku pamięci."
    )
    args = parser.parse_args()

    if args.check:
        return tryb_check()

    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "BŁĄD: brak OPENROUTER_API_KEY.\n"
            'Ustaw:  export OPENROUTER_API_KEY="sk-or-v1-..."\n'
            "Albo uruchom walidację offline:  python agent_react_langgraph.py --check",
            file=sys.stderr,
        )
        return 2

    graph = build_graph()
    wynik = run_thread_test(graph, thread_id=args.thread_id)

    print("\n" + "=" * 78)
    print("PODSUMOWANIE TESTU")
    print("=" * 78)
    print(json.dumps(wynik, ensure_ascii=False, indent=2))
    print("\n" + "=" * 78)
    print("ODPOWIEDŹ 1 (pogoda):")
    print(wynik["odpowiedz_1"])
    print("\nODPOWIEDŹ 2 (ubiór - z użyciem pamięci):")
    print(wynik["odpowiedz_2"])
    print("=" * 78)

    if wynik["uzyto_narzedzia"] and wynik["pamiec_ok"]:
        print("\n✅ TEST ZALICZONY: narzędzie wywołane, pamięć wątku zachowana.")
        return 0
    print("\n❌ TEST NIEZALICZONY - sprawdź logi powyżej.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
