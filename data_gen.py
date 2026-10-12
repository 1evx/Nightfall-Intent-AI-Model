"""
Synthetic dataset generator for the survivor intent classifier.

Each category is built from named "families" -- one family is one sentence
pattern (e.g. "go to the {room}") instantiated across every applicable slot
value plus light augmentation (name prefix, casing, punctuation). Splitting
into train/val/test happens at the FAMILY level, not the individual-example
level: this is what makes val/test a genuine test of generalizing to unseen
phrasing rather than just unseen slot values, which -- since rooms/tasks/
stock types are closed enumerable classes anyway -- need to appear in every
split regardless.

Labeling is not done by re-implementing SurvivorCommandExecutor's regex
in Python; each family is hand-written to be category-pure (its filler
words never accidentally contain a higher-precedence trigger phrase), with
the matching-order edge cases from that file worked through by hand in
comments below where they matter (e.g. "take me to the kitchen" falling
through Give/SelfServe to Room because no item keyword is present).
"""
import json
import random

import vocab as V

SEED = 20260821
rng = random.Random(SEED)

ITEM_PHRASES = {
    "Food": ["food", "some food", "a meal", "some meals", "rations", "a snack", "some rations"],
    "Pill": ["pills", "some pills", "a pill", "some pill", "the pills"],
    "Medicine": ["medicine", "some medicine", "meds", "medication", "a medkit", "some meds"],
}
BARE_ITEM_WORDS = {
    "Food": ["food", "meals", "rations", "snacks"],
    "Pill": ["pills"],
    "Medicine": ["medicine", "meds", "medication"],
}

NAME_PREFIXES = ["{name}, ", "hey {name}, ", "{name} -- "]
POLITE_PREFIXES = ["", "", "", "please ", "can you ", "could you ", "hey, "]
PUNCT_SUFFIXES = ["", "", "", ".", "?", "!"]


def augment(text, allow_question=True, n=2):
    """Yield n augmented variants of one rendered template string."""
    out = []
    for _ in range(n):
        t = text
        if rng.random() < 0.25:
            t = rng.choice(NAME_PREFIXES).format(name=rng.choice(V.SURVIVOR_NAMES)) + t
        elif rng.random() < 0.35:
            t = rng.choice(POLITE_PREFIXES) + t
        suffix_pool = PUNCT_SUFFIXES if allow_question else [s for s in PUNCT_SUFFIXES if s != "?"]
        t = t + rng.choice(suffix_pool)
        style = rng.random()
        if style < 0.5:
            t = t[0].upper() + t[1:] if t else t
        elif style < 0.7:
            t = t.lower()
        out.append(t)
    return out


class Builder:
    """Collects (text, category, room, task, stock) examples grouped into
    named families, for later family-level train/val/test splitting."""

    def __init__(self):
        self.families = {}  # (category, family_name) -> list[example dict]

    def add(self, category, family_name, texts, room=None, task=None, stock=None):
        key = (category, family_name)
        bucket = self.families.setdefault(key, [])
        for t in texts:
            bucket.append({"text": t, "category": category, "room": room, "task": task, "stock": stock})

    def family(self, category, family_name, template, room=None, task=None, stock=None, allow_question=True, n_aug=6):
        variants = augment(template, allow_question=allow_question, n=n_aug)
        self.add(category, family_name, variants, room=room, task=task, stock=stock)


def build():
    b = Builder()

    # --- Follow / Stay / Resume: no slots ---
    follow = [
        "follow me", "come with me", "follow", "stick with me", "come along with me",
        "walk with me", "tag along", "stay close to me", "come along", "let's go together",
        "follow me around", "walk beside me", "keep up with me", "come this way with me",
        "stick close and follow me", "walk together with me", "walk alongside me",
        "don't leave my side", "shadow me", "escort me", "join me", "accompany me",
        "move with me", "stay by my side", "hang with me", "be my shadow",
        "walk right behind me", "come stand next to me", "trail behind me",
        "stick with me from now on", "come with me for now",
    ]
    for i, t in enumerate(follow):
        b.family("Follow", f"follow_{i}", t)

    stay = [
        "stay here", "stay", "wait here", "hold position", "stay put", "don't move",
        "wait right there", "hold there", "stay where you are", "just wait here",
        "stay still", "hold your position", "don't go anywhere", "wait for me right here",
        "don't leave this spot", "remain here", "hold your ground", "wait for my signal",
        "don't budge", "sit tight", "stay right there", "remain in place",
        "camp here", "settle here", "wait right here for now", "stay put for now",
        "don't move from that spot", "hold up right there",
    ]
    for i, t in enumerate(stay):
        b.family("Stay", f"stay_{i}", t)

    resume = [
        "resume", "resume your routine", "go back to your routine", "forget that",
        "never mind, go back to normal", "back to your normal routine", "carry on as usual",
        "go back to what you were doing", "resume normal routine", "as you were",
        "drop that and go back to normal", "cancel that, resume your routine",
        "go back to your day-to-day routine", "release from that and resume routine",
        "go back to your usual routine", "return to your normal tasks",
        "pick up your routine again", "go about your business", "back to your day job",
        "continue your routine", "return to normal", "let go of that order",
        "release the current order", "back to your usual self",
        "go back to what you normally do", "you're free, resume routine",
    ]
    for i, t in enumerate(resume):
        b.family("Resume", f"resume_{i}", t)

    # --- Give / Return: item slot (Food/Pill/Medicine) ---
    # "Give" covers both plain hand-off phrasing ("give me some food") and
    # what used to be a separate "SelfServe" category ("grab some food
    # yourself") -- SurvivorCommandExecutor resolves both to the exact same
    # SelfServeItem action, so they're trained as one category now (see
    # vocab.py's CATEGORIES comment). Keeping both template lists (instead of
    # collapsing to one) still matters for training data: the model needs to
    # recognize the item request regardless of whether a "yourself"/"central
    # hall" qualifier is present or not.
    give_templates = [
        "give {item}", "give you some {item}", "here, take some {item}", "take this {item}",
        "grab some {item}", "here's {item} for you", "take a {item}", "have some {item}",
        "grab {item} for the house", "take {item} for home", "here, have this {item}",
        "i'm giving you {item}", "take some {item} from me", "grab a {item}",
        "here, i brought you {item}", "take this, it's {item}",
        "grab some {item} yourself", "go get {item} yourself", "take {item} yourself",
        "grab {item} from central hall", "get {item} from central hall yourself",
        "go fetch {item} yourself", "grab yourself some {item}",
        "take yourself some {item} from central hall", "go grab {item} from the central hub yourself",
        "fetch some {item} from central hall yourself",
    ]
    return_templates = [
        "give me back the {item}", "give back the {item}", "take back the {item}",
        "return the {item}", "hand back the {item}", "bring back the {item}",
        "i want the {item} back", "can i have the {item} back", "take that {item} back",
        "return that {item} to me", "hand that {item} back to me", "i need the {item} back",
    ]
    for stock, items in ITEM_PHRASES.items():
        for i, tmpl in enumerate(give_templates):
            item = items[i % len(items)]
            b.family("Give", f"give_{stock}_{i}", tmpl.format(item=item), stock=stock)
        for i, tmpl in enumerate(return_templates):
            item = items[i % len(items)]
            b.family("Return", f"return_{stock}_{i}", tmpl.format(item=item), stock=stock)

    # --- Location / Status / Suggestion / ResourceAll: no slots, phrase-list + paraphrase ---
    location = V.LOCATION_QUERY_PHRASES + [
        "where do you stand right now", "can you tell me where you are",
        "what's your current location", "where might you be",
        "do you know where you are right now", "where on earth are you",
        "tell me where you are", "what's your position",
        "point me to your spot", "which room is that", "i need to know where you are",
        "say where you're standing", "give me your coordinates",
        "what part of the house are you in", "let me know where you are",
        "can you say where you're at",
    ]
    for i, t in enumerate(location):
        b.family("Location", f"loc_{i}", t)

    status = V.STATUS_QUERY_PHRASES + [
        "how's it going for you", "feeling alright", "everything okay with you",
        "how are things", "you holding up okay", "give me a status update",
        "how are you holding up", "everything alright over there",
    ]
    for i, t in enumerate(status):
        b.family("Status", f"status_{i}", t)

    suggestion = V.SUGGESTION_QUERY_PHRASES + [
        "what should we tackle next", "any ideas what to do next",
        "what do you think i should do", "give me some direction",
        "what's next on the list", "what should i prioritize",
        "any thoughts on what's next", "point me in a direction",
    ]
    for i, t in enumerate(suggestion):
        b.family("Suggestion", f"suggest_{i}", t)

    resource_all = V.RESOURCE_QUERY_PHRASES + [
        "how are we on supplies", "give me a supply update",
        "what resources do we have left", "run me through our stock",
        "how are we doing on stock", "give me the supply rundown",
    ]
    for i, t in enumerate(resource_all):
        b.family("ResourceAll", f"resall_{i}", t)

    # --- ResourceOne: "how many/how much" + bare item word ---
    resource_one_templates = [
        "how many {item} do we have", "how much {item} do we have left",
        "how many {item} are left", "do we have enough {item}",
        "check how much {item} we have", "how's our {item} supply",
        "how much {item} is left", "how many {item} do we have left",
    ]
    for stock, items in BARE_ITEM_WORDS.items():
        for i, tmpl in enumerate(resource_one_templates):
            item = items[i % len(items)]
            b.family("ResourceOne", f"resone_{stock}_{i}", tmpl.format(item=item), stock=stock)

    # --- Task: label + aliases, wrapped in generic frames ---
    task_frames = [
        "go to {alias}", "can you go to {alias}", "head to {alias}", "time for {alias}",
        "go do {alias}", "{alias}", "let's do {alias} today", "off to {alias}",
        "go work the {alias}", "send them to {alias}",
    ]
    for label, aliases in V.TASKS.items():
        for i, alias in enumerate(aliases):
            for j, frame in enumerate(task_frames):
                # Thin the cross product, but always keep the two most
                # natural/common frames ("go to {alias}" and the bare alias)
                # for every single alias -- skipping them for some aliases
                # (the old "(i + j) % 3 != 0" thinning did, whenever i%3!=0)
                # meant an alias like "Outpost" or "Lumberyard" had *no*
                # "go to X" training example at all, so real commands like
                # "go to the outpost" fell through to a Room guess instead.
                if j not in (0, 5) and (i + j) % 2 != 0:
                    continue
                # n_aug=4 (not the family() default of 6): this cross product
                # alone would otherwise dwarf every other category now that
                # it's no longer thinned down to 1-in-3, skewing the category
                # head's prior toward Task. The place/extra phrasings below
                # stay at the full default since they're hand-curated, not a
                # combinatorial blow-up.
                b.family("Task", f"task_{label}_{i}_{j}", frame.format(alias=alias), task=label, n_aug=4)

    # Outpost/Fishing Hut/Lumberyard/Front Porch are also registered rooms
    # (see vocab.py's _TASK_ALIAS_ROOMS), so players naturally phrase them
    # the way they'd phrase any other room -- "go to THE lumberyard" -- which
    # task_frames above never generates (it has no article at all). Without
    # this, that exact phrasing was untrained and the model fell back on
    # whatever it had learned from Room's "go to the {room}" family instead,
    # guessing a real room name rather than the task. Re-use the same
    # "the"-bearing phrasing style as Room's room_frames further down, just
    # labeled as the Task it actually is.
    TASK_ALIAS_ROOM_FOR = {
        "Prepare Venture": "Outpost", "Fishing": "Fishing Hut",
        "Woodcutting": "Lumberyard", "Resting": "Front Porch",
    }
    place_task_frames = [
        "go to the {alias}", "can you go to the {alias}", "head to the {alias}",
        "walk to the {alias}", "please go to the {alias}", "go check out the {alias}",
        "move to the {alias}", "go work the {alias}", "send them to the {alias}",
        "go to {alias}", "{alias}",
    ]
    for label, room_alias in TASK_ALIAS_ROOM_FOR.items():
        for j, frame in enumerate(place_task_frames):
            b.family("Task", f"task_{label}_place_{j}", frame.format(alias=room_alias), task=label)

    # Woodcutting and Prepare Venture were the two worst-covered tasks (only
    # 3-4 short aliases each vs. Resting's 9), and the alias+frame generator
    # above never produces the way people actually phrase them out loud --
    # "cut/chop some wood" (which shares the "verb + some + noun" shape with
    # Give's "grab some {item}" and was getting misread as Give), or "go to
    # the lumberyard"/"the outpost" (patched above). These hand-written
    # phrasings close that gap directly instead of hoping the frame
    # cross-product eventually covers it.
    woodcutting_extra = [
        "cut wood", "cut some wood", "go cut wood", "go cut some wood",
        "chop wood", "chop some wood", "go chop wood", "go chop some wood",
        "let's go chop some wood", "let's cut some wood", "time to chop wood",
        "time to cut some wood", "get some wood chopped", "go get some wood cut",
        "start woodcutting", "let's do some woodcutting", "go do some woodcutting",
        "it's time for some woodcutting", "go chop some firewood", "cut some firewood",
        "chop some firewood", "go cut some firewood", "send them to cut wood",
        "can you go chop some wood", "go chop wood for a while",
    ]
    for i, t in enumerate(woodcutting_extra):
        b.family("Task", f"task_Woodcutting_extra_{i}", t, task="Woodcutting")

    venture_extra = [
        "prepare venture", "prepare for the venture", "let's prepare for the venture",
        "get ready for the venture", "time to venture out", "start the venture",
        "prep the venture", "prep for the venture", "ready the venture",
        "send them on the venture", "go on the venture", "let's get the venture ready",
        "get the venture ready", "time to prepare the venture", "go prepare for the venture",
        "it's venture time", "let's venture out", "head out on the venture",
        "can you prepare for the venture", "go get ready for the venture",
    ]
    for i, t in enumerate(venture_extra):
        b.family("Task", f"task_PrepareVenture_extra_{i}", t, task="Prepare Venture")

    # --- Room: 14 non-task-alias rooms, generic frames ---
    room_frames = [
        "go to the {room}", "can you go to the {room}", "head to the {room}",
        "go wander the {room}", "go to {room}", "walk to the {room}",
        "please go to the {room}", "go check out the {room}", "move to the {room}",
        # Edge case: "take"/"grab" present but no item keyword -> falls through
        # Give/SelfServe checks (no stock type found) and lands on Room.
        "take me to the {room}", "can you take me to the {room}",
    ]
    for i, room in enumerate(V.ROOM_ONLY):
        for j, frame in enumerate(room_frames):
            if (i + j) % 2 != 0:
                continue
            b.family("Room", f"room_{i}_{j}", frame.format(room=room), room=room)

    # --- None: small talk + hard negatives that share words with triggers
    # but don't satisfy the real matching rules (verified by hand against
    # SurvivorCommandExecutor.ExecuteTextCommand's precedence). ---
    none_examples = [
        "nice weather today", "how's the weather", "tell me a joke",
        "what's your favorite color", "i had a long day",
        "let's talk about something else", "arrest roommate",
        "let's presume the worst", "did you take the trash out",
        "grab a coffee sometime", "don't give up on me",
        "the resistance is growing", "did you hear the news", "i'm bored",
        "sing me a song", "what's for dinner tonight", "do you like music",
        "hello there", "hi", "good morning", "thanks for everything",
        "you're doing great", "i appreciate you", "see you later", "goodnight",
        "that's a great story", "i'm glad you're safe", "this place is a mess",
        "did you sleep well", "i miss the old days", "what a day",
        "we should celebrate soon", "i trust your judgment",
    ]
    for i, t in enumerate(none_examples):
        b.family("None", f"none_{i}", t)

    return b


def split_families(b, val_frac=0.1, test_frac=0.1):
    by_cat = {}
    for (cat, fam), examples in b.families.items():
        by_cat.setdefault(cat, []).append((fam, examples))

    train, val, test = [], [], []
    for cat, fams in by_cat.items():
        fams = list(fams)
        rng.shuffle(fams)
        n = len(fams)
        n_val = max(1, round(n * val_frac)) if n >= 5 else 0
        n_test = max(1, round(n * test_frac)) if n >= 10 else 0
        val_fams = fams[:n_val]
        test_fams = fams[n_val:n_val + n_test]
        train_fams = fams[n_val + n_test:]
        for _, examples in train_fams:
            train.extend(examples)
        for _, examples in val_fams:
            val.extend(examples)
        for _, examples in test_fams:
            test.extend(examples)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    b = build()
    train, val, test = split_families(b)
    write_jsonl("data/train.jsonl", train)
    write_jsonl("data/val.jsonl", val)
    write_jsonl("data/test.jsonl", test)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    from collections import Counter
    print("category counts (train):", Counter(e["category"] for e in train))


if __name__ == "__main__":
    main()
