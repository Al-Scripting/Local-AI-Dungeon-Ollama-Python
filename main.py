import json
import os
import sys
import textwrap
import re
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import requests  # pip install requests

# ---------------- Constants / Paths ----------------

RULES_PATH = "rules.json"
GM_PROMPT_PATH = os.path.join("prompts", "gm.txt")
TRANSCRIPT_PATH = os.path.join("samples", "transcript.txt")
SAVE_PATH = "save.json"

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.environ.get("AI_DUNGEON_MODEL", "gemma3:latest")

# Context / UX
HISTORY_TURNS = 6
DEFAULT_TIMEOUT_SEC = 180


# ---------------- Utility ----------------

def load_rules() -> Dict[str, Any]:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    # Defaults: keep combat stable if missing from rules.json
    rules.setdefault("ENEMIES", {
        "wolf": {
            "name": "Wolf", "hp": 7, "atk_min": 1, "atk_max": 2,
            "flee_player_chance": 0.7, "on_defeat_flag": "wolf_defeated", "on_defeat_loot": []
        },
        "bandit": {
            "name": "Bandit", "hp": 9, "atk_min": 1, "atk_max": 3,
            "flee_player_chance": 0.55, "on_defeat_flag": "bandit_defeated", "on_defeat_loot": ["key_a1"]
        },
        "skeletal_guard": {
            "name": "Skeletal Guard", "hp": 10, "atk_min": 2, "atk_max": 3,
            "flee_player_chance": 0.45, "on_defeat_flag": "guard_cleared", "on_defeat_loot": ["bone_charm"]
        }
    })
    rules.setdefault("LOCATION_DESCRIPTIONS", {})
    rules.setdefault("MAP", {})
    rules.setdefault("LOCKS", {})
    return rules


def load_gm_prompt() -> str:
    with open(GM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def init_state(rules: Dict[str, Any]) -> Dict[str, Any]:
    start = deepcopy(rules.get("START", {}))
    return {
        "location": start.get("location", "Unknown"),
        "inventory": list(start.get("inventory", [])),
        "flags": dict(start.get("flags", {})),
        "hp": int(start.get("hp", 10)),
        "turns": 0,
        # "battle": {"enemy_id": str, "hp": int}
    }


def save_state(state: Dict[str, Any]) -> None:
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("[Saved]")


def load_state() -> Optional[Dict[str, Any]]:
    if not os.path.exists(SAVE_PATH):
        print("[No save file found]")
        return None
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    print("[Loaded]")
    return state


def ensure_transcript() -> None:
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    if not os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write("# Sample session transcript\n")


def append_transcript(entry: str) -> None:
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# ---------------- Command validation ----------------

def _tokenize(s: str) -> List[str]:
    return s.strip().lower().split()


def is_valid_command(user_input: str, commands: List[str]) -> bool:
    ui = user_input.strip().lower()
    if not ui:
        return False
    if ui in commands:
        return True

    ui_parts = _tokenize(ui)
    for cmd in commands:
        p_parts = _tokenize(cmd)
        if not p_parts:
            continue
        fixed_count = sum(1 for p in p_parts if not (p.startswith("<") and p.endswith(">")))
        if fixed_count > len(ui_parts):
            continue
        i = j = 0
        ok = True
        while i < len(p_parts) and j < len(ui_parts):
            p_tok = p_parts[i]
            if p_tok.startswith("<") and p_tok.endswith(">"):
                if i == len(p_parts) - 1:
                    j = len(ui_parts)
                    i += 1
                else:
                    j += 1
                    i += 1
            else:
                if ui_parts[j] != p_tok:
                    ok = False
                    break
                i += 1
                j += 1
        if ok and i == len(p_parts) and j == len(ui_parts):
            return True
    return False


# ---------------- LLM interaction ----------------

def build_messages(
    gm_prompt: str,
    rules: Dict[str, Any],
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
    user_input: str,
) -> List[Dict[str, str]]:
    recent = history[-HISTORY_TURNS:]
    recent_summary = [{"player": t["user"], "gm": t["gm"]} for t in recent]
    return [
        {"role": "system", "content": gm_prompt},
        {"role": "system", "content": "RULES:\n" + json.dumps(rules, ensure_ascii=False)},
        {"role": "system", "content": "CURRENT_STATE:\n" + json.dumps(state, ensure_ascii=False)},
        {"role": "system", "content": "LAST_TURNS:\n" + json.dumps(recent_summary, ensure_ascii=False)},
        {"role": "user", "content": user_input},
    ]


def call_ollama(messages: List[Dict[str, str]]) -> Optional[str]:
    payload = {"model": MODEL_NAME, "messages": messages, "stream": False, "format": "json"}
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ReadTimeout:
        print("[Engine timeout] Model took too long to respond.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Engine error] {e}")
        return None


# ---------------- Rules enforcement & helpers ----------------
def has_item(state, item_name: str) -> bool:
    return item_name in (state.get("inventory") or [])


def enforce_max_paragraphs(narration: str, max_paragraphs: int) -> str:
    paras = [p for p in narration.split("\n") if p.strip()]
    return "\n".join(paras[:max_paragraphs])


def is_locked_destination(dest: str, rules: Dict[str, Any], state: Dict[str, Any]) -> bool:
    lock = rules.get("LOCKS", {}).get(dest)
    if not lock:
        return False
    # Treat having the required key item as satisfying the lock, too.
    if lock == "have_key_a1" and any("key_a1" == it for it in state.get("inventory", [])):
        state["flags"]["have_key_a1"] = True
        return False
    return not state["flags"].get(lock, False)


def find_direction_in_text(text: str) -> Optional[str]:
    t = text.lower()
    if any(w in t for w in ["north", " ahead", " forward", " go north", " move north", " n "]):
        return "north"
    if any(w in t for w in ["south", " back", " backward", " backwards", " go south", " move south", " s "]):
        return "south"
    if any(w in t for w in ["east", " right", " go east", " move east", " e "]):
        return "east"
    if any(w in t for w in ["west", " left", " go west", " move west", " w "]):
        return "west"
    return None


def engine_move(rules: Dict[str, Any], state: Dict[str, Any], user_input: str) -> str:
    """Engine-owned movement with MAP + LOCKS, returns narration string."""
    cur = state.get("location")
    dirn = find_direction_in_text(user_input)
    if not dirn:
        # List exits
        neighbors = rules.get("MAP", {}).get(cur, {})
        if not neighbors:
            return "There are no obvious exits from here."
        exits = []
        for d, dest in neighbors.items():
            tag = " (locked)" if is_locked_destination(dest, rules, state) else ""
            exits.append(f"{d} → {dest}{tag}")
        return "You can’t go that way from here. Try: " + " | ".join(exits)

    neighbors = rules.get("MAP", {}).get(cur, {})
    dest = neighbors.get(dirn)
    if not dest:
        return "You can’t go that way from here."

    if is_locked_destination(dest, rules, state):
        # Specific guidance for known locks
        if dest == "Ancient Gate":
            return "The Ancient Gate is sealed by runes. You’ll need a key to unlock it."
        if dest == "Crown Chamber":
            return "The vault beyond is sealed. Open the gate first."
        return f"The way to {dest} is blocked."

    # Perform move
    state["location"] = dest
    # Optional: auto-open flag when stepping through unlocked gate into Crown Chamber path
    if cur == "Ruins Entrance" and dest == "Ancient Gate" and state["flags"].get("have_key_a1"):
        state["flags"]["have_gate_opened"] = True

    # Encounter will be handled later in loop
    desc = rules.get("LOCATION_DESCRIPTIONS", {}).get(dest)
    if desc:
        return desc
    return f"You move {dirn} to {dest}."


def apply_state_changes(
    rules: Dict[str, Any],
    state: Dict[str, Any],
    state_changes: List[Dict[str, Any]],
    last_command: str,
) -> Dict[str, Any]:
    """Apply only legal atoms; ignore LLM movement & combat; enforce inventory & flags."""
    inv_limit = int(rules.get("INVENTORY_LIMIT", 5))

    for atom in state_changes:
        if not isinstance(atom, dict):
            continue
        op = atom.get("op")

        if op == "move_to":
            # Engine owns movement; ignore LLM teleports to prevent desync.
            print("[Blocked] LLM move_to ignored (engine owns movement).")
            continue

        elif op == "add_item":
            item = atom.get("item")
            if not item:
                continue
            if len(state["inventory"]) >= inv_limit:
                print("[Blocked] Inventory is full.")
                continue
            if item not in state["inventory"]:
                state["inventory"].append(item)
                # Auto-flag key possession
                if item == "key_a1":
                    state["flags"]["have_key_a1"] = True

        elif op == "remove_item":
            item = atom.get("item")
            if item in state["inventory"]:
                state["inventory"].remove(item)

        elif op == "set_flag":
            flag = atom.get("flag")
            if flag:
                state["flags"][flag] = True

        elif op == "clear_flag":
            flag = atom.get("flag")
            if flag and flag in state["flags"]:
                state["flags"].pop(flag, None)

        elif op == "hp_delta":
            # Combat is engine-owned; ignore LLM hp changes
            print("[Blocked] hp_delta ignored (engine owns combat).")
            continue

    return state


def check_end_conditions(rules: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    end = rules.get("END_CONDITIONS", {})
    win_flags = end.get("WIN_ALL_FLAGS", [])
    lose_flags = end.get("LOSE_ANY_FLAGS", [])
    max_turns = end.get("MAX_TURNS")

    for f in lose_flags:
        if state["flags"].get(f, False):
            return "lose", f"Defeat. ({f} triggered.)"
    if max_turns is not None and state["turns"] >= max_turns:
        return "lose", "You have run out of time."
    if win_flags and all(state["flags"].get(f, False) for f in win_flags):
        return "win", "You have completed your quest and secured the Crown."
    return None, None


# ---------------- Parsing helpers ----------------

def parse_gm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()

    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", s, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if '"narration":' in s and '"state_change":' in s:
        nm = re.search(r'"narration"\s*:\s*"(.*?)"', s, re.S)
        sm = re.search(r'"state_change"\s*:\s*(\[[\s\S]*?\])', s)
        if nm:
            narration = nm.group(1)
            leak = narration.find(',"state_change"')
            if leak != -1:
                narration = narration[:leak].rstrip()
            state_change = []
            if sm:
                try:
                    state_change = json.loads(sm.group(1))
                except Exception:
                    state_change = []
            return {"narration": narration, "state_change": state_change}

    return None


def sanitize_narration(narration: str, max_paragraphs: int) -> str:
    bad_idx = narration.find(',"state_change"')
    if bad_idx != -1:
        narration = narration[:bad_idx].rstrip()
    return enforce_max_paragraphs(narration or "", max_paragraphs)


# ---------------- Battle system (engine-owned) ----------------

def in_battle(state: Dict[str, Any]) -> bool:
    return state.get("flags", {}).get("in_battle", False) is True and "battle" in state


def get_enemy_stats(rules: Dict[str, Any], enemy_id: str) -> Dict[str, Any]:
    e = rules.get("ENEMIES", {}).get(enemy_id, {})
    return {
        "name": e.get("name", enemy_id.title()),
        "hp": int(e.get("hp", 8)),
        "atk_min": int(e.get("atk_min", 1)),
        "atk_max": int(e.get("atk_max", 3)),
        "flee_player_chance": float(e.get("flee_player_chance", 0.5)),
        "on_defeat_flag": e.get("on_defeat_flag"),
        "on_defeat_loot": list(e.get("on_defeat_loot", [])),
    }


def start_battle(state: Dict[str, Any], enemy_id: str, rules: Dict[str, Any]) -> str:
    stats = get_enemy_stats(rules, enemy_id)
    state["flags"]["in_battle"] = True
    state["battle"] = {"enemy_id": enemy_id, "hp": int(stats["hp"])}
    return f"A {stats['name']} appears! (HP {stats['hp']}) You are in combat: attack | defend | flee"


def end_battle(state: Dict[str, Any], rules: Dict[str, Any], victory: bool) -> str:
    enemy_id = state.get("battle", {}).get("enemy_id", "foe")
    stats = get_enemy_stats(rules, enemy_id)
    state["flags"].pop("in_battle", None)
    state.pop("battle", None)
    if victory:
        if stats.get("on_defeat_flag"):
            state["flags"][stats["on_defeat_flag"]] = True
        for item in stats.get("on_defeat_loot", []):
            if item not in state["inventory"]:
                state["inventory"].append(item)
                if item == "key_a1":
                    state["flags"]["have_key_a1"] = True
        return f"You defeated the {stats['name']}."
    return f"You escaped the {stats['name']}."


def player_weapon_bonus(state: Dict[str, Any]) -> int:
    for it in state.get("inventory", []):
        if "sword" in it.lower():
            return 1
    return 0


def battle_turn_attack(state: Dict[str, Any], rules: Dict[str, Any]) -> str:
    b = state["battle"]
    enemy_id = b["enemy_id"]
    stats = get_enemy_stats(rules, enemy_id)

    dmg_player = random.randint(2, 4) + player_weapon_bonus(state)
    b["hp"] -= dmg_player
    narration = [f"You strike the {stats['name']} for {dmg_player} damage."]

    if b["hp"] <= 0:
        narration.append(end_battle(state, rules, victory=True))
        return " ".join(narration)

    dmg_enemy = random.randint(stats["atk_min"], stats["atk_max"])
    state["hp"] = max(0, state["hp"] - dmg_enemy)
    narration.append(f"The {stats['name']} hits you for {dmg_enemy} damage. (Your HP: {state['hp']})")
    if state["hp"] <= 0:
        state["flags"]["hp_zero"] = True
    return " ".join(narration)


def battle_turn_defend(state: Dict[str, Any], rules: Dict[str, Any]) -> str:
    b = state["battle"]
    enemy_id = b["enemy_id"]
    stats = get_enemy_stats(rules, enemy_id)
    dmg_enemy = max(0, random.randint(stats["atk_min"], stats["atk_max"]) - 1)
    state["hp"] = max(0, state["hp"] - dmg_enemy)
    if state["hp"] <= 0:
        state["flags"]["hp_zero"] = True
    return f"You brace for impact, reducing the blow. The {stats['name']} deals {dmg_enemy} damage. (Your HP: {state['hp']})"


def battle_turn_flee(state: Dict[str, Any], rules: Dict[str, Any]) -> str:
    b = state["battle"]
    enemy_id = b["enemy_id"]
    stats = get_enemy_stats(rules, enemy_id)
    if random.random() < stats["flee_player_chance"]:
        return end_battle(state, rules, victory=False)
    dmg_enemy = random.randint(stats["atk_min"], stats["atk_max"])
    state["hp"] = max(0, state["hp"] - dmg_enemy)
    if state["hp"] <= 0:
        state["flags"]["hp_zero"] = True
    return f"You try to flee but stumble! The {stats['name']} hits you for {dmg_enemy}. (Your HP: {state['hp']})"


# ---------------- Encounters ----------------

def maybe_trigger_encounter(
    rules: Dict[str, Any],
    state: Dict[str, Any],
    last_location: Optional[str],
    last_command: str,
) -> Optional[str]:
    if in_battle(state):
        return None
    loc = state.get("location")
    entries = rules.get("ENCOUNTERS", {}).get(loc, [])
    if not entries:
        return None
    moved = (state.get("location") != last_location)
    if not moved:
        return None
    for e in entries:
        try:
            if random.random() < float(e.get("chance", 0.0)):
                return start_battle(state, str(e.get("id", "foe")).strip(), rules)
        except Exception:
            continue
    return None


def battle_limited_command(user_input: str) -> bool:
    allowed = ("attack", "defend", "flee", "inventory", "help", "save", "load", "quit")
    ui = user_input.strip().lower()
    return any(ui == a or ui.startswith(a + " ") for a in allowed)


# ---------------- I/O helpers ----------------

def print_status_and_commands(state: Dict[str, Any]) -> None:
    print(f"\n[Location: {state['location']}] HP: {state['hp']} Turn: {state['turns']}")
    print("Commands: look | move <direction> | take <item> | search | use <item> on <target> | talk | talk <npc> | "
          "attack | defend | flee | inventory | help | save | load | quit")
    if in_battle(state):
        b = state["battle"]
        enemy_id = b["enemy_id"]
        hp = b["hp"]
        name = rules_global.get("ENEMIES", {}).get(enemy_id, {}).get("name", enemy_id.title())
        print(f"[ENCOUNTER] {name} (HP {hp}) engages you! (attack | defend | flee)")


def print_help(rules: Dict[str, Any]) -> None:
    print("Available commands:")
    for c in rules.get("COMMANDS", []):
        print(f" - {c}")


def print_inventory(state: Dict[str, Any]) -> None:
    inv = state.get("inventory", [])
    if not inv:
        print("Inventory is empty.")
    else:
        print("Inventory:")
        for item in inv:
            print(f" - {item}")


# ---------------- Action handlers (engine-owned where needed) ----------------

def handle_use(rules: Dict[str, Any], state: Dict[str, Any], user_input: str) -> Optional[str]:
    """Engine resolves key → gate. Returns narration if handled; None to fall back to LLM."""
    t = user_input.lower()
    if "use" in t and "key" in t and ("gate" in t or "lock" in t):
        # Require key in inventory
        if "key_a1" not in state.get("inventory", []):
            return "You don’t have a suitable key."
        # Must be near the gate
        if state.get("location") not in ("Ruins Entrance", "Ancient Gate"):
            return "There’s nothing here that fits that key."
        state["flags"]["have_key_a1"] = True
        state["flags"]["have_gate_opened"] = True
        return "You turn the key in the Ancient Gate’s lock. Runes dim and stone grinds aside—the way north is open."
    return None


def handle_take(rules: Dict[str, Any], state: Dict[str, Any], user_input: str) -> Optional[str]:
    """Engine enforces crown pickup only in Crown Chamber."""
    t = user_input.lower()
    if t.startswith("take ") and "crown" in t:
        if state.get("location") != "Crown Chamber":
            return "You reach for the Crown, but it isn’t here."
        # Add crown and flag
        if "crown" not in state["inventory"]:
            state["inventory"].append("crown")
        state["flags"]["crown_recovered"] = True
        return "You lift the Crown from its dais. Power hums through the chamber."
    return None


def maybe_mark_returned(state: Dict[str, Any]) -> None:
    if state.get("location") == "Village Square" and state.get("flags", {}).get("crown_recovered"):
        state["flags"]["returned_to_village"] = True


# ---------------- Main loop ----------------

def main() -> None:
    global rules_global
    if not os.path.exists(RULES_PATH):
        print("rules.json not found.")
        sys.exit(1)

    rules = load_rules()
    rules_global = rules  # for encounter banner names
    gm_prompt = load_gm_prompt()
    state = init_state(rules)
    history: List[Dict[str, Any]] = []

    # Intro
    prologue = rules.get("PROLOGUE")
    if prologue:
        print(textwrap.fill(prologue, 80))
        print()
    quest_intro = rules.get("QUEST", {}).get("intro")
    if quest_intro:
        print(textwrap.fill(quest_intro, 80))
        print()

    ensure_transcript()

    while True:
        # Win/lose check
        maybe_mark_returned(state)
        status, msg = check_end_conditions(rules, state)
        if status:
            print(f"\n*** {msg} ***")
            break

        print_status_and_commands(state)
        user_input = input("> ").strip()
        low = user_input.lower()

        # Meta (no LLM)
        if low == "help":
            print_help(rules)
            continue
        if low == "inventory":
            print_inventory(state)
            continue
        if low == "save":
            save_state(state)
            continue
        if low == "load":
            loaded = load_state()
            if loaded:
                state = loaded
            continue
        # Engine-owned 'search' shortcut at Ruins Entrance for guaranteed progression
        # --- Deterministic SEARCH (engine-owned progression)
        low = user_input.strip().lower()
        if low == "search":
            loc = state.get("location")
            # Guaranteed progression: reveal the key exactly once at Ruins Entrance
            if loc == "Ruins Entrance" and not state["flags"].get("have_key_a1"):
                if "key_a1" not in state["inventory"]:
                    state["inventory"].append("key_a1")
                state["flags"]["have_key_a1"] = True
                state["flags"]["ruins_entrance_searched"] = True
                state["turns"] += 1
                msg = "Behind a cracked pillar you uncover a small iron key, cold to the touch."
                print("\n" + textwrap.fill(msg, 80))
                append_transcript(json.dumps(
                    {"player": user_input, "engine_search_award": "key_a1", "state_after": state},
                    ensure_ascii=False))
                continue
            else:
                # Let the GM narrate harmless searching elsewhere (no engine changes)
                pass
        # --- Hard-ground TAKE to prevent duplicates and GM contradictions
        if low.startswith("take "):
            wanted = low.replace("take ", "", 1).strip()

            # Simple alias normalization (edit as you like)
            alias_map = {
                "iron key": "key_a1",
                "key": "key_a1",
                "the crown": "crown",
                "crown": "crown",
            }
            wanted_id = alias_map.get(wanted, wanted)

            if has_item(state, wanted_id):
                print("\nYou already have that.")
                state["turns"] += 1
                append_transcript(json.dumps(
                    {"player": user_input, "engine_take_already_owned": wanted_id, "state_after": state},
                    ensure_ascii=False))
                continue
            # Else: let the GM handle proposing {"op":"add_item"} if it’s actually obtainable
            # End of hard-ground TAKE

        if low == "quit":
            print("Goodbye.")
            break

        # Battle phase: engine only
        if in_battle(state):
            if not battle_limited_command(user_input):
                print("You are in combat. Use: attack | defend | flee (or inventory/help).")
                append_transcript(f"PLAYER: {user_input}\nBLOCKED_IN_BATTLE")
                continue
            if low.startswith("attack"):
                narration = battle_turn_attack(state, rules)
            elif low.startswith("defend"):
                narration = battle_turn_defend(state, rules)
            elif low.startswith("flee"):
                narration = battle_turn_flee(state, rules)
            else:
                narration = "Hold your ground. (Use attack | defend | flee.)"
            state["turns"] += 1
            print()
            print(textwrap.fill(narration, 80))
            append_transcript(json.dumps({"player": user_input, "engine_battle": True, "state_after": state}, ensure_ascii=False))
            continue

        # Engine-owned actions before LLM
        if low.startswith("move "):
            last_loc = state.get("location")
            narration = engine_move(rules, state, user_input)
            # Turn advances on movement attempt
            state["turns"] += 1
            # Encounter after genuine move
            enc_msg = maybe_trigger_encounter(rules, state, last_loc, user_input)
            if enc_msg:
                print(enc_msg)
            print()
            print(textwrap.fill(narration, 80))
            append_transcript(json.dumps({"player": user_input, "engine_move": True, "state_after": state}, ensure_ascii=False))
            continue

        # Engine-enforced special uses / takes
        used = handle_use(rules, state, user_input)
        if used:
            state["turns"] += 1
            print()
            print(textwrap.fill(used, 80))
            append_transcript(json.dumps({"player": user_input, "engine_use": True, "state_after": state}, ensure_ascii=False))
            continue

        took = handle_take(rules, state, user_input)
        if took:
            state["turns"] += 1
            print()
            print(textwrap.fill(took, 80))
            append_transcript(json.dumps({"player": user_input, "engine_take": True, "state_after": state}, ensure_ascii=False))
            continue

        # Block combat verbs outside encounters
        if low.startswith("attack") or low.startswith("defend") or low.startswith("flee"):
            print("There is no immediate threat here. Try: look | move <direction> | talk")
            append_transcript(f"PLAYER: {user_input}\nBLOCKED_NO_ENCOUNTER")
            continue

        # Validate against COMMANDS
        allowed_cmds = [c.lower() for c in rules.get("COMMANDS", [])]
        if not is_valid_command(user_input, allowed_cmds):
            print("Unknown or illegal command. Type 'help' for valid commands.")
            continue

        # LLM call for flavor/interaction (non-movement/non-combat)
        messages = build_messages(gm_prompt, rules, state, history, user_input)
        raw = call_ollama(messages)
        if not raw:
            append_transcript(f"PLAYER: {user_input}\nGM_ERROR_OR_TIMEOUT")
            continue

        gm_reply = parse_gm_json(raw)
        if not isinstance(gm_reply, dict):
            print("[Invalid GM JSON, ignoring turn.]")
            append_transcript(f"PLAYER: {user_input}\nGM_INVALID: {raw}")
            continue

        narration = sanitize_narration(
            gm_reply.get("narration", "") or "",
            int(rules.get("MAX_PARAGRAPHS", 2))
        )
        state_changes = gm_reply.get("state_change", []) or []

        # Apply legal (non-movement, non-combat) changes
        last_location = state.get("location")
        state = apply_state_changes(rules, state, state_changes, user_input)

        # After talk/look/etc., do not trigger encounters unless we actually moved
        # Turn advances
        state["turns"] += 1

        # If the user typed 'look', prefer canonical location description
        if low == "look":
            desc_map = rules.get("LOCATION_DESCRIPTIONS", {})
            forced = desc_map.get(state.get("location", ""), "").strip()
            if forced:
                narration = forced

        # Print narration
        if narration:
            print()
            print(textwrap.fill(narration, 80))

        # Transcript & history
        entry = {"player": user_input, "gm": gm_reply, "state_after": state}
        append_transcript(json.dumps(entry, ensure_ascii=False))
        history.append({"user": user_input, "gm": gm_reply})


if __name__ == "__main__":
    main()
# End of main.py