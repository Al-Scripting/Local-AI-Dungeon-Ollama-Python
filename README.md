---

# 🧠 Local AI Dungeon (Ollama + Python)

A **fully local, privacy-friendly text-adventure engine** powered by a **Gemma 3 (or any Ollama) model**.
This project demonstrates how a lightweight Python game loop can integrate with a local LLM to create an interactive story world governed entirely by JSON-based rules.

The AI narrates the world and proposes game state changes, while the Python engine enforces logic, progression, and win/lose conditions — ensuring deterministic and replayable adventures.

---

## 🚀 Features

* **Offline AI Game Master** — all narration generated locally via Ollama
* **Rule-Driven Gameplay** — world constraints, quests, and locks defined in `rules.json`
* **Dynamic State Management** — persistent location, inventory, HP, and flags
* **Enforced Logic Layer** — prevents illegal actions or rule-breaking outputs from the model
* **Transcript Logging** — records each turn as structured JSON for debugging or replay
* **Save/Load System** — resume sessions via `save.json`
* **Model Agnostic** — works with `gemma3:latest`, `llama3.1`, `mistral`, or any local chat model

---

## 🧩 How It Works

The engine runs a simple turn-based loop:

1. **Player Input** → You type a command (e.g., `move Ancient Gate`, `take sigil_key`).
2. **Rules + Context Sent to Model** → The engine passes:

   * `prompts/gm.txt` (system prompt defining JSON schema and constraints)
   * `rules.json` (world logic)
   * Current game state
   * Last few turns
3. **Model Reply (JSON)** → The LLM returns:

   ```json
   {
     "narration": "You approach the Ancient Gate...",
     "state_change": [{"op": "move_to", "location": "Ancient Gate"}]
   }
   ```
4. **Engine Enforcement** → The Python code:

   * Validates and parses the JSON
   * Blocks illegal moves or inventory overflow
   * Updates state and checks end conditions
5. **Narration Displayed** → The player sees the narration and continues.

If all quest flags are achieved, the game announces victory. If HP reaches zero or a lose condition triggers, it ends gracefully.

---

## 📁 Project Structure

```
ai-dungeon/
│
├── main.py                 # Core engine: loop, Ollama integration, rule enforcement
├── rules.json              # Defines world logic, commands, quest, and endings
├── README.md               # Project documentation
│
├── prompts/
│   └── gm.txt              # System prompt defining model behavior and JSON format
│
├── samples/
│   └── transcript.txt      # Example or live session log
│
└── save.json               # Auto-generated save file (after using 'save' command)
```

---

## ⚙️ Installation & Setup

### 1. Requirements

* **Python 3.9+**
* **Ollama** installed and running in the background
  → [https://ollama.com/download](https://ollama.com/download)

### 2. Clone or Create Folder

```bash
git clone https://github.com/<your-username>/ai-dungeon.git
cd ai-dungeon
```

### 3. Pull the Model

```bash
ollama pull gemma3:latest
```

*(You may also use `llama3.1` or any other chat-capable model.)*

### 4. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\Activate.ps1      # Windows PowerShell
```

### 5. Install Dependencies

```bash
pip install requests
```

---

## ▶️ Running the Game

Make sure Ollama is running in the background.

```bash
python main.py
```

You’ll see:

```text
The village elder begs you to recover the stolen Crown...

[Location: Village Square] HP: 10 Turn: 0
> 
```

Try commands like:

```text
> look
> move Ancient Gate
> take sigil_key
> inventory
> save
> load
> quit
```

---

## 🧠 Using a Different Model

You can specify another Ollama model at runtime:

```bash
AI_DUNGEON_MODEL="gemma3:latest" python main.py     # macOS / Linux
$env:AI_DUNGEON_MODEL="gemma3:latest"; python main.py  # Windows PowerShell
```

---

## 🧱 Game Logic Overview (`rules.json`)

| Section             | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| **COMMANDS**        | Defines all valid player actions (prevents illegal input). |
| **LOCKS**           | Maps locked areas to required flags.                       |
| **QUEST**           | Specifies the main quest name, intro, and goal flag.       |
| **END_CONDITIONS**  | Defines win/lose flags and max turns.                      |
| **START**           | Initial state: location, inventory, HP, flags.             |
| **INVENTORY_LIMIT** | Caps the number of items.                                  |

Example:

```json
"LOCKS": { "Ancient Gate": "have_key_a1" },
"END_CONDITIONS": {
  "WIN_ALL_FLAGS": ["crown_recovered", "returned_to_village"],
  "LOSE_ANY_FLAGS": ["hp_zero"],
  "MAX_TURNS": 50
}
```

---

## 💾 Saving and Loading

* `save` → writes the current state to `save.json`
* `load` → restores the saved state
* `quit` → exits the game safely

All player interactions are automatically logged in `samples/transcript.txt` for analysis or replay.

---

## 🧪 Example Session

See `samples/transcript.txt` for a 7-turn sample run demonstrating:

* Movement blocked by locked gate
* Picking up an item to unlock it
* Fulfilling the quest
* Win detection triggered when quest flags complete

---

## 🧭 Extending the Game

You can create your own adventures by editing **`rules.json`** and **`prompts/gm.txt`**:

* Change the **QUEST** section for a new storyline
* Add new **LOCATIONS**, **ITEMS**, or **LOCKS**
* Adjust the **COMMANDS** list to match your desired verbs
* Modify the **system prompt** for different narrative styles (mystery, sci-fi, horror, etc.)

No Python code needs to be changed — the engine automatically adapts to new JSON rules.

---

## 📚 Educational Purpose

This project is designed to demonstrate:

* Integrating **local LLMs** as controllable narrative agents
* Building **rule-driven simulations** with a constrained state machine
* Managing **LLM outputs safely** through post-processing and validation

It’s an ideal example for courses or workshops on **AI-assisted interactive fiction, prompt engineering, or LLM integration**.

---

## 🏁 License

MIT License © 2025
Developed for educational and experimental purposes — run locally, modify freely, and explore how rules and reasoning interact with generative storytelling.

---
