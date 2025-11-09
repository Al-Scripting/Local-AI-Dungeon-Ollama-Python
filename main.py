
import json
import os
import sys
import textwrap
from copy import deepcopy

import requests  # pip install requests

RULES_PATH = "rules.json"
GM_PROMPT_PATH = os.path.join("prompts", "gm.txt")
TRANSCRIPT_PATH = os.path.join("samples", "transcript.txt")
SAVE_PATH = "save.json"

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.environ.get("AI_DUNGEON_MODEL", "gemma3:latest")


# ---------------- Utility ----------------


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gm_prompt():
    with open(GM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def init_state(rules):
    start = deepcopy(rules["START"])
    state = {
        "location": start.get("location", "Unknown"),
        "inventory": start.get("inventory", []),
        "flags": start.get("flags", {}),
        "hp": start.get("hp", 10),
        "turns": 0,
    }
    return state


def save_state(state):
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("[Saved]")


def load_state():
    if not os.path.exists(SAVE_PATH):
        print("[No save file found]")
        return None
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    print("[Loaded]")
    return state


def append_transcript(entry):
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# ---------------- Command validation ----------------


def is_valid_command(user_input, rules):
    user_input = user_input.strip().lower()
    if not user_input:
        return False

    parts = user_input.split()
    verb = parts[0]

    # Find matching command template (e.g., "move <direction>")
    for command_template in rules.get("COMMANDS", []):
        template_parts = command_template.split()
        if verb == template_parts[0]:
            # If it's a move command, check if the second part is a valid direction
            if verb == "move" and len(parts) > 1:
                direction = parts[1]
                if direction in rules.get("DIRECTIONS", []):
                    return True
            # For other commands, a simple verb match is enough for now
            elif len(parts) == 1:
                return True

    return False


# ---------------- LLM interaction ----------------


def build_messages(gm_prompt, rules, state, history, user_input):
    """
    Keep total context small: only last few turns.
    history: list of dicts { "user": str, "gm": dict }
    """
    # include last N turns
    N = 6
    recent = history[-N:]

    recent_summary = []
    for turn in recent:
        recent_summary.append({"player": turn["user"], "gm": turn["gm"]})

    return [
        {"role": "system", "content": gm_prompt},
        {"role": "system", "content": "RULES:\n" + json.dumps(rules)},
        {"role": "system", "content": "CURRENT_STATE:\n" + json.dumps(state)},
        {
            "role": "system",
            "content": "LAST_TURNS:\n" + json.dumps(recent_summary),
        },
        {"role": "user", "content": user_input},
    ]


def call_ollama(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "format": "json",  # helps push the model toward strict JSON
    }
    resp = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    return content


# ---------------- Rules enforcement ----------------


def enforce_max_paragraphs(narration, max_paragraphs):
    paras = [p for p in narration.split("\n") if p.strip()]
    return "\n".join(paras[:max_paragraphs])


def is_locked_destination(dest, rules, state):
    lock = rules.get("LOCKS", {}).get(dest)
    if not lock:
        return False
    return not state["flags"].get(lock, False)


def apply_state_changes(rules, state, state_changes, user_input):
    verb = user_input.lower().split()[0] if user_input else ""
    if verb == "move":
        direction = user_input.lower().split()[1]
        current_location_name = state["location"]
        current_location = rules.get("LOCATIONS", {}).get(current_location_name, {})
        exits = current_location.get("exits", {})

        if direction in exits:
            destination = exits[direction]
            if not is_locked_destination(destination, rules, state):
                state["location"] = destination
            else:
                print(f"[Blocked] The way to {destination} is locked.")
        else:
            print("[Blocked] You can't go that way.")
        return state

    inv_limit = int(rules.get("INVENTORY_LIMIT", 5))

    for atom in state_changes:
        if not isinstance(atom, dict):
            continue

        op = atom.get("op")
        if op == "move_to":
            dest = atom.get("location")
            if not dest:
                continue
            if is_locked_destination(dest, rules, state):
                print(f"[Blocked] The way to {dest} is locked.")
                continue
            state["location"] = dest

        elif op == "add_item":
            item = atom.get("item")
            if not item:
                continue
            if len(state["inventory"]) >= inv_limit:
                print("[Blocked] Inventory is full.")
                continue
            if item not in state["inventory"]:
                state["inventory"].append(item)

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
            delta = atom.get("delta", 0)
            try:
                delta = int(delta)
            except (TypeError, ValueError):
                continue
            state["hp"] = int(state.get("hp", 0)) + delta
            if state["hp"] <= 0:
                state["hp"] = 0
                state["flags"]["hp_zero"] = True

        # ignore unknown ops silently (engine is source of truth)

    return state


def check_end_conditions(rules, state):
    end = rules.get("END_CONDITIONS", {})
    win_flags = end.get("WIN_ALL_FLAGS", [])
    lose_flags = end.get("LOSE_ANY_FLAGS", [])
    max_turns = end.get("MAX_TURNS")

    # lose: flags
    for f in lose_flags:
        if state["flags"].get(f, False):
            return "lose", f"Defeat. ({f} triggered.)"

    # lose: turn limit
    if max_turns is not None and state["turns"] >= max_turns:
        return "lose", "You have run out of time."

    # win: all flags
    if win_flags and all(state["flags"].get(f, False) for f in win_flags):
        return "win", "You have completed your quest and secured the Crown."

    return None, None


# ---------------- Main loop ----------------


def print_help(rules):
    print("Available commands:")
    for c in rules["COMMANDS"]:
        print(f" - {c}")


def print_inventory(state):
    inv = state["inventory"]
    if not inv:
        print("Inventory is empty.")
    else:
        print("Inventory:")
        for item in inv:
            print(f" - {item}")


def main():
    if not os.path.exists(RULES_PATH):
        print("rules.json not found.")
        sys.exit(1)

    rules = load_rules()
    gm_prompt = load_gm_prompt()

    state = init_state(rules)
    history = []

    # intro
    quest_intro = rules.get("QUEST", {}).get("intro")
    if quest_intro:
        print(textwrap.fill(quest_intro, 80))
        print()

    # ensure transcript file exists
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write("# Sample session transcript\n")

    while True:
        status, msg = check_end_conditions(rules, state)
        if status == "win":
            print(f"\n*** {msg} ***")
            break
        elif status == "lose":
            print(f"\n*** {msg} ***")
            break

        print(
            f"\n[Location: {state['location']}] HP: {state['hp']} Turn: {state['turns']}"
        )
        user_input = input("> ").strip()

        # meta commands (no LLM call)
        if user_input.lower() == "help":
            print_help(rules)
            continue

        if user_input.lower() == "inventory":
            print_inventory(state)
            continue

        if user_input.lower() == "save":
            save_state(state)
            continue

        if user_input.lower() == "load":
            loaded = load_state()
            if loaded:
                state = loaded
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        # validate command against COMMANDS
        if not is_valid_command(user_input, rules):
            print("Unknown or illegal command. Type 'help' for valid commands.")
            continue

        # build context and call local LLM
        messages = build_messages(gm_prompt, rules, state, history, user_input)
        try:
            raw = call_ollama(messages)
        except Exception as e:
            print(f"[Engine error] {e}")
            break

        # parse GM JSON
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                # handle cases like ```json { ... }
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()
            gm_reply = json.loads(raw)

        except json.JSONDecodeError:
            print("[Invalid GM JSON, ignoring turn.]")
            append_transcript(f"PLAYER: {user_input}\nGM_INVALID: {raw}")
            continue

        narration = gm_reply.get("narration", "")
        state_changes = gm_reply.get("state_change", [])

        # enforce MAX_PARAGRAPHS
        max_p = int(rules.get("MAX_PARAGRAPHS", 2))
        narration = enforce_max_paragraphs(narration, max_p)

        # apply legal state changes
        state = apply_state_changes(rules, state, state_changes, user_input)

        # increment turns
        state["turns"] += 1

        # print narration
        if narration:
            print()
            print(textwrap.fill(narration, 80))

        # log to transcript
        entry = {"player": user_input, "gm": gm_reply, "state_after": state}
        append_transcript(json.dumps(entry, ensure_ascii=False))

        # add to in-memory history (for next prompts)
        history.append({"user": user_input, "gm": gm_reply})


if __name__ == "__main__":
    main()
