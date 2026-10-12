"""
Single source of truth for the survivor command grammar. Transcribed from
SurvivorCommandExecutor.cs (trigger-phrase lists, matching order) and the
game's room/task/stock design. Keep this in sync with the actual Unity scene
data (RoomZone.roomName, LocationTaskDefinition.taskLabel/commandAliases,
PlayerStock.StockType) -- the C# side treats those as authoritative, and this
file's ROOMS/TASKS/STOCK_TYPES lists must match them or the trained model's
slot heads will predict values the game doesn't recognize.
"""

# Fixed order -- each name must exactly match a member of
# SurvivorIntentClassifier.IntentCategory (C#); Enum.TryParse there maps by
# name, not by this array's index, so the order itself isn't load-bearing.
#
# "Give" and "SelfServe" used to be trained as two separate categories
# ("give some food" vs. "grab some food yourself"), but SurvivorCommandExecutor
# has treated them as the exact same action (SelfServeItem) for a while now --
# see its GiveTriggerPhrases handling and the Give/SelfServe case in
# ExecuteTextCommandAsync. Training them as two classes just gave the model
# two near-identical phrasings to confuse against each other for a
# distinction the game no longer acts on, so they're merged into one "Give"
# category here; the C# enum keeps its unused SelfServe member for the LLM
# backend, which still classifies against the full enum independently.
CATEGORIES = [
    "Follow", "Stay", "Resume", "Give", "Return",
    "Task", "Room", "Location", "Status", "Suggestion",
    "ResourceAll", "ResourceOne", "None",
]

NO_MATCH = "NoMatch"

# All 18 registered rooms. Note: Outpost, Front Porch, Lumberyard, and
# Fishing Hut double as task aliases below -- SurvivorCommandExecutor checks
# FindTaskSiteInText BEFORE roomRegistry.FindInText, so a command mentioning
# one of those four names is a Task match, not a Room match. ROOM_ONLY excludes
# them for Room-category example generation; ROOMS keeps the full 18 for the
# model's room output head (a Room-category prediction could in principle
# still land on any of them if the caller's dynamic `choices` list changes).
ROOMS = [
    "Guest Room", "Dining Room", "Shower Room", "Kitchen", "Meeting Room",
    "Rest Room", "Entrance", "Front Porch", "Storage Room", "Outpost",
    "Library", "Lumberyard", "Bath Room", "Second Floor Sub Room",
    "Central Hall", "Fishing Hut", "Second Floor", "Master Room",
]

_TASK_ALIAS_ROOMS = {"Outpost", "Front Porch", "Lumberyard", "Fishing Hut"}
ROOM_ONLY = [r for r in ROOMS if r not in _TASK_ALIAS_ROOMS]

# label -> alias phrases (label itself is also a valid trigger, matched the
# same way TextMatch.MatchesPhrase(text, site.TaskLabel) does in the C# code).
TASKS = {
    "Prepare Venture": ["Prepare Venture", "Outpost", "Venture"],
    "Fishing": ["Fishing", "Go Fishing", "Fishing Hut", "Catch Fish", "Fish"],
    "Woodcutting": ["Woodcutting", "Lumberyard", "Chop Wood", "Cut Wood"],
    "Resting": [
        "Resting", "Go Resting", "Rest", "Go Rest", "Front Porch", "Home",
        "House", "Come Back", "Come Home", "Back Home",
    ],
}
TASK_LABELS = list(TASKS.keys())

# Matches C# PlayerStock.StockType's declared order (None, Food, Pill,
# Medicine) -- not load-bearing (mapping is by name string, not ordinal) but
# kept aligned for readability.
STOCK_TYPES = ["None", "Food", "Pill", "Medicine"]
STOCK_KEYWORDS = {
    "Food": ["food", "meal", "meals", "rations", "snack"],
    "Pill": ["pill", "pills"],
    "Medicine": ["medicine", "meds", "medication", "medkit"],
}

# --- Trigger-phrase lists, verbatim from SurvivorCommandExecutor.cs ---

LOCATION_QUERY_PHRASES = [
    "where are you", "where you at", "where're you", "where you are",
    "what room are you", "which room are you", "your location", "current location",
]

STATUS_QUERY_PHRASES = [
    "are you doing good", "are you doing okay", "are you doing alright",
    "are you okay", "are you alright", "you doing good", "you doing okay",
    "you okay", "you alright", "how are you", "how you doing",
    "how are you feeling", "how you feeling", "status report", "your status",
]

SUGGESTION_QUERY_PHRASES = [
    "what can you do for me", "what can you do", "what could you do",
    "what do you suggest", "what would you suggest", "any suggestions",
    "any suggestion", "what should i do", "what should we do",
    "what can i do", "what can we do", "any recommendations",
    "any recommendation", "recommend anything", "give me advice",
    "any advice", "any tips", "got any tips", "any tip",
    "what are my options", "what are the options", "where should i send you",
    "where should you go", "help me out", "what's the plan", "whats the plan",
]

RESOURCE_QUERY_PHRASES = [
    "how many resource", "how many resources", "how much resource",
    "how much resources", "resource count", "how many supplies",
    "how much supplies", "how are our supplies", "how's our supplies",
    "hows our supplies", "check supplies", "check our supplies",
    "what do we have", "what do we got", "inventory check",
    "how's our stock", "hows our stock", "what's in stock", "whats in stock",
    "how much stock",
]

GIVE_TRIGGER_PHRASES = ["give", "grab", "take"]
RETURN_TRIGGER_PHRASES = ["return", "give back", "give me back", "take back", "hand back", "bring back"]
SELF_SERVE_QUALIFIER_PHRASES = ["yourself", "central hall", "central hub"]
RESOURCE_QUANTITY_TRIGGER_PHRASES = ["how many", "how much"]

# Fake survivor first names, used only to build "<Name>, ..." prefixes so the
# model learns to ignore a leading name the same way roster.FindInText/the
# caller already strips it out before this classifier ever sees the text in
# production (SurvivorCommandExecutor resolves survivor by name separately).
SURVIVOR_NAMES = ["Sarah", "Marcus", "Elena", "Tom", "Priya", "Jonah", "Ava", "Deshawn"]
