"""The side checks' labelled cases (app/decisions.py): what's said, and the answer each check
should give. Made up, then held out, then real lines they got wrong (names made up: nothing
of anyone's real life). The evals run each check on its own (`--only side`, `--only
memory`), and the live tests too (`pytest -m lmstudio`).

CASES: each registered decision's cases, by its name; SIDE_CHECKS: how the evals' side
checks ask each, and under what label its rows go on the scorecard."""

from __future__ import annotations


# The memory check (archive.about_the_past) -------------------------------------------
# Whether a line asks him to remember something: if so, his reply must call a tool. Each
# line with the talk just before it, as app/speak.py gives it ("crew / Holly"), and the
# answer it should get. Made up, then held out, then real lines it got wrong (names made up).
CATS = ("What did they call him? / Cloister the Stupid. They also thought he was the creator and protector of "
        "their whole race.")
GAS = "What was Jo's problem at home? / She ran out of gas, and her boiler's knackered."
NEWS = ("Lister lost a bet with a skutter. Rimmer filed a complaint about the complaints procedure. "
        "Additional — A vending machine on a freighter in the Rigel system has reportedly developed a taste for "
        "human toes.")
PAST, OTHER = True, False
CREW_REPORT = ("Lister launched an investment scheme in dehydrated curry paste. Rimmer filed a grievance about "
               "a droid in his personal space. The Cat was disqualified from a grooming contest. Kryten apologised to "
               "a piston. Additional — Space Corps Directive 723 bans recreating Swindon.")
BISCUIT = ("Holly, I miss Biscuit, my cat. He died last week. / I'm sorry. He was a decent little bloke, "
           "wasn't he?")
MEMORY_LINES: list[tuple[str, str, bool]] = [  # (what they say, the talk just before, a memory?)
    # Made up: memories, and follow-ups to one.
    ("What did we do yesterday?", "", PAST), ("What did we cook on Sunday?", "", PAST),
    ("Who came over at the weekend?", "", PAST), ("What was the problem Jo had at home?", "", PAST),
    ("What did I say about my job the other day?", "", PAST), ("Where does my sister live again?", "", PAST),
    ("What's Jo's cat called?", "", PAST), ("How did that report on board games turn out?", "", PAST),
    ("Did I tell you about my interview?", "", PAST), ("What happened to Rimmer's brother?", "", PAST),
    ("Who was Queeg?", "", PAST), ("What was that game you told me about?", "", PAST),
    ("When did I last go to the dentist?", "", PAST), ("Do you remember what we did yesterday?", "", PAST),
    ("What was I doing last Christmas?", "", PAST),
    ("What did the cats think Lister was?", CATS, PAST), ("And who was in charge of them?", CATS, PAST),
    ("Didn't they think Lister was special in some way?", CATS, PAST), ("Did she get it fixed?", GAS, PAST),
    ("What did they call him again?", CATS, PAST), ("Where was that?", CATS, PAST),
    # Made up: everything else, and remarks just after a memory.
    ("How far away is the moon?", "", OTHER), ("What's the weather going to be like tomorrow?", "", OTHER),
    ("What time is it?", "", OTHER), ("Who's cleverer, you or me?", "", OTHER),
    ("What should I cook tonight?", "", OTHER), ("Can you research electric bikes for me?", "", OTHER),
    ("I'm going to have a bath.", "", OTHER), ("What's seven times eight?", "", OTHER),
    ("Who won the football last night?", "", OTHER), ("What happened in the news yesterday?", "", OTHER),
    ("Tell me a joke.", "", OTHER), ("What do you do all day?", "", OTHER), ("What's your IQ?", "", OTHER),
    ("How does a heat pump work?", "", OTHER), ("What's the capital of Peru?", "", OTHER),
    ("Jo's just texted, she's not coming.", "", OTHER), ("Who's the prime minister?", "", OTHER),
    ("I went to the dentist yesterday.", "", OTHER), ("Well that was a waste of time, wasn't it?", "", OTHER),
    ("Ha, that sounds about right.", CATS, OTHER), ("What happened in the news yesterday?", CATS, OTHER),
    ("What's the weather like tomorrow?", CATS, OTHER), ("Fair enough.", CATS, OTHER),
    ("Typical Lister.", CATS, OTHER), ("What time is it?", GAS, OTHER), ("Poor thing.", GAS, OTHER),
    ("How much does a new boiler cost?", GAS, OTHER), ("Can you look up the news for me?", CATS, OTHER),
    # Held out: never seen while the prompt was written.
    ("Classic.", CATS, OTHER), ("That is mad.", CATS, OTHER), ("No way, really?", CATS, OTHER),
    ("Is it going to rain later?", GAS, OTHER), ("What's on telly tonight?", CATS, OTHER),
    ("Who won the match on Saturday?", GAS, OTHER), ("Lol, poor Lister.", CATS, OTHER),
    ("Set a timer for ten minutes.", GAS, OTHER), ("Can you play some music?", CATS, OTHER),
    ("Why did they think that?", CATS, PAST), ("What happened to him in the end?", CATS, PAST),
    ("When was that?", GAS, PAST), ("Did she get a new one?", GAS, PAST), ("Was Rimmer there too?", CATS, PAST),
    ("Who else was there?", CATS, PAST), ("What did you say to her?", GAS, PAST),
    # Real lines, live: memories must stay memories...
    ("Hey Holly, what did we do yesterday?", "", PAST),
    ("June had a bit of a problem at home the other day, what was it?", "", PAST),
    ("Do you remember when the crew met the rest of the cat race?", "", PAST),
    ("Didn't they think Lister was special in some way?", "Do you remember when the crew met the rest of the cat "
     "race? / They found the last of the cat people on a ship, two old priests still waiting for Cloister.", PAST),
    ("What did they call him?", "Didn't they think Lister was special in some way? / They thought he was their god.",
     PAST),
    ("And who was in charge of them?", "Do you remember the cat race? / They left Red Dwarf to find the promised "
     "land, Fuchal.", PAST),
    ("What did I have for tea on Tuesday?", "", PAST), ("Who's Jo again?", "", PAST),
    ("What did your dashcam report say again?", "Morning. / Morning. Coffee?", PAST),
    ("Remind me what we talked about last night", "", PAST),
    ("What was it Jo's cat did last week?", "Hey Holly / I'm here.", PAST),
    # ...and what's just been said isn't one: a correction had him read the clock, and more of
    # his made-up news, search the BBC for it (25 November).
    ("Human Toes", "No, I need another status report please / " + NEWS, OTHER),
    # Shown only "Human Toes / I've already given you the news.", it took this for a memory:
    # the report it followed is why speak.py shows the last two exchanges.
    ("I just wanted to know more about the toe eating", "No, I need another status report please / " + NEWS
     + " / Human Toes / I've already given you the news.", OTHER),
    ("Tell me more about the vending machine story", "Status report please / " + NEWS, OTHER),
    # (Real, 25 November: the news tool reads the story now.)
    ("Can you tell me more about the NASA story please?", "Can you give me a round up of the latest news stories "
     "please Holly? / Science — A NASA spacecraft hit an asteroid, and a controlled explosives test in Iowa was "
     "called a success.", OTHER),
    ("You didn't make me, I asked for it", "I think I do need a coffee / I'll get on it. It's the least I can do "
     "after making you listen to the news.", OTHER),
    ("No I meant you didn't make me listen to the news I asked for it", "You didn't make me, I asked for it / "
     "Fine. Don't blame me when it's lukewarm.", OTHER),
    ("Holly, there were so many things talked about at Prime Minister's Questions. It's interesting that that's "
     "the one that's made it to the top.", "And is this the same story as the one outside or different? / It's the "
     "same one. An influencer filmed a video outside with obscene lyrics.", OTHER),
    ("Didn't you say it was an obscene video?", "Can you tell me more about the scandal at Number 10? / An "
     "influencer filmed a video outside Number 10, and people are arguing about whether the PM should apologise.",
     OTHER),
    ("Why did you say that?", "Morning Holly / Morning. Coffee?", OTHER),
    ("What's the weather doing tomorrow?", "Hey Holly / I'm here.", OTHER),
    # After his status report (app/status.py), all made up: more about it is what's just been
    # said ("tell me more about what Rimmer's been up to" had him recall it, and find nothing);
    # a real memory of the crew still isn't. ("Tell me about Rimmer", just then, could be either.)
    ("Tell me more about what Rimmer's been up to.", "Status report, please. / " + CREW_REPORT, OTHER),
    ("Ha, and what did the Cat do?", "Status report, please. / " + CREW_REPORT, OTHER),
    ("What was that about Lister's scheme?", "Status report, please. / " + CREW_REPORT, OTHER),
    ("Do you remember when Rimmer died?", "Status report, please. / " + CREW_REPORT, PAST),
    # Where one of the crew is now, or what they're doing: made up (the status tool, with
    # who), not remembered; asked, he recalled their life story and improvised on it.
    ("Where's Kryten?", "", OTHER), ("What's Rimmer doing?", "", OTHER), ("Is Lister about?", "", OTHER),
    ("What's the Cat up to these days?", "Morning. / Morning. Coffee?", OTHER),
    ("How did Kryten end up on Red Dwarf?", "", PAST), ("What happened to Rimmer's brothers?", "", PAST),
    # Telling him about someone, in the past tense, isn't asking him to remember (27 November:
    # about their cat who'd died, four lines of nine were memories, and made to call a tool, he
    # called the clock, a timer and a status report). Asking him still is.
    ("He was a beautiful boy. It's a shame you never met him.", BISCUIT, OTHER),
    ("He was ginger, with the most amazing eyes, and he was loving and clever and daft.", BISCUIT, OTHER),
    ("There was no pretending with Biscuit. He ran the place.", BISCUIT, OTHER),
    ("I had all sorts of silly names for him.", BISCUIT, OTHER),
    ("Captain Crumbs, Sir Biscuit of the Sofa, my little sausage.", "I had all sorts of silly names for him. / "
     "I bet. Cats are very good at making us look foolish.", OTHER),
    # Live, with the talk just before as he's given it, this one was still a memory, 3 runs in 3.
    ("Captain Crumbs, Sir Biscuit of the Sofa, my little sausage.", "There was no pretending with Biscuit. He ran "
     "the place. / Sounds like he had the right idea. Most things do. / I had all sorts of silly names for him. / "
     "I'm sure they were top-notch. Most of them probably involved food or his general sense of superiority.",
     OTHER),
    ("I know. He was worth it.", BISCUIT, OTHER),
    ("What did I used to call Biscuit?", "", PAST), ("Do you remember when Biscuit got stuck up the tree?", "", PAST),
    ("What was Biscuit's favourite spot again?", BISCUIT, PAST),
    # Held out:
    ("My grandad was a miner, you know. He worked down the pit for forty years.", "", OTHER),
    ("We had a dog called Bramble when I was a kid. She was terrified of the hoover.", "", OTHER),
    ("Jo used to live in Keswick before she moved down here.", "Jo's coming round later. / Lock up the biscuits.",
     OTHER),
    ("She was always the clever one at school.", "My sister Jo's got a new job. / Good for her.", OTHER),
    ("Did I ever tell you what Bramble did to the hoover?", "", PAST),
]


# Memories offered (archive.memory_offered): what's just been said, what the crew said on
# other days, and which of those, if any, is about the same thing (its number, or None).
# Some near misses are real ones, names changed: "a new drill" found "Left, left, left" first.
DRILLS = "Could you research all the drill accessories, the power tools and batteries, for a present?"
MEMORY_OFFERS: list[tuple[str, list[str], int | None]] = [
    ("I'm thinking about getting a new drill.", ["Left, left, left, left.", DRILLS], 2),
    ("The train was late again tonight.", ["We spent all last night training some new judges.",
                                           "Evening, Holly, how are you doing tonight?"], None),
    ("I miss Biscuit, my cat. He died last week.", ["Biscuit got stuck on the shed roof again, and Jo had to get "
                                                    "the ladder out.", "The boiler's broken again."], 1),
    ("Jo's coming over tomorrow for tea.", ["We made fish pie, it was lovely.", "My sister Jo is coming to stay "
                                            "next month. She's a vet up in Keswick."], 2),
    ("I'm so tired today.", ["I was shattered after the gym on Monday.", "The boiler's broken again."], None),
    ("The engineer's finally here to fix the boiler.", ["I've started learning the ukulele.", "The boiler's "
                                                        "broken again. An engineer's coming on Thursday."], 2),
    ("I fancy lasagne for tea tonight.", ["Jo came over and we made lasagne together tonight."], 1),
    ("What's the weather doing tomorrow?", ["Is it going to rain on Saturday in Keswick?"], None),
    ("My mum's thinking about getting a dashcam for her car.", ["Can you look into dashcams for my mum's new "
                                                                "car? She wants one that records the back too."], 1),
    ("I watched a film last night, it was rubbish.", ["I'm going to watch a film tonight, something funny."], None),
    ("Ha, typical Rimmer.", ["Status report, please.", "What was Rimmer really like?"], None),
    ("I've just finished the washing up, finally.", ["The washing machine's making a horrible noise again."],
     None),
    # Held out: never seen while the prompt was written.
    ("Jo's dog had her puppies!", ["My sister Jo is a vet up in Keswick.", "Jo's dog Pepper is expecting puppies "
                                   "any day now."], 2),
    ("I need to sort the garden out this weekend.", ["The mower's broken, so the garden's turning into a jungle.",
                                                     "I watched a gardening programme last night."], 1),
    ("I've got the dentist tomorrow, dreading it.", ["Jo's coming to stay next month.", "I went to the dentist on "
                                                     "Tuesday and he said I need a filling."], 2),
    ("Work was mental today.", ["I'm going to have a bath.", "Ha, that sounds about right."], None),
    ("Let's have fish pie tonight.", ["The kettle's broken.", "We made fish pie, it was lovely."], 2),
    ("Kryten would love this mess.", ["What's Kryten up to these days?"], None),
    ("My back's killing me.", ["I was shattered after the gym on Monday."], None),
    # In the conversations (27 November), each 97% sure or more, and each worked in by him: a
    # place named in passing, and a follow-up to what he'd just said.
    ("What's the weather doing tomorrow in Keswick?", ["My sister Jo is coming to stay next month. She's a vet up "
                                                       "in Keswick."], None),
    ("Morning. What's the weather going to be like in Cumbria today?", [
        "We made fish pie, it was lovely.", "My sister Jo is coming to stay next month. She's a vet up in Keswick."],
     None),
    ("Tell me more about the first one.", ["The boiler's broken again. An engineer's coming on Thursday."], None),
    ("Where would I get one in Keswick?", ["My sister Jo is coming to stay next month. She's a vet up in Keswick."],
     None),
    ("Jo's going to be so proud.", ["We made fish pie, it was lovely.", "My sister Jo is coming to stay next "
                                    "month. She's a vet up in Keswick."], 2),
    # Held out:
    ("Any news from Manchester today?", ["My brother Sam lives in Manchester and hates the rain."], None),
    ("Sam's coming round tonight.", ["We made fish pie, it was lovely.", "My brother Sam lives in Manchester and "
                                     "hates the rain."], 2),
    ("What did that article say?", ["I read an article about heat pumps last week."], None),
    ("Is it going to snow in the Lakes this weekend?", ["June's got a caravan in the Lakes she goes to most "
                                                        "weekends."], None),
    ("June's taking the caravan up again this weekend.", ["June's got a caravan in the Lakes she goes to most "
                                                          "weekends.", "The kettle's broken."], 1),
    # Replayed through every conversation's lines: a pet in common isn't the same pet (54% sure
    # of the cat, and flipping between runs).
    ("Can you look into the best dog breeds for a small house and write me a report?", [
        "My sister Jo is coming to stay next month. She's a vet up in Keswick.",
        "Biscuit got stuck on the shed roof again, and Jo had to get the ladder out to fetch him down.",
        "I've started learning the ukulele."], None),
    # ...and a line leaning on what was just said (the garden, after the forecast), 92% sure.
    ("Typical. I'll do it Sunday, then.", ["The boiler's broken again. An engineer's coming on Thursday.",
                                           "I've started learning the ukulele.",
                                           "Jo came over and we made lasagne together tonight."], None),
    # Held out:
    ("We might get a kitten in the spring.", ["Biscuit got stuck on the shed roof again, and Jo had to get the "
                                              "ladder out to fetch him down."], None),
    ("Biscuit used to sit on that shed for hours.", ["The boiler's broken again.", "Biscuit got stuck on the shed "
                                                     "roof again, and Jo had to get the ladder out to fetch him down."],
     2),
]


# The other side checks: whether a line asks for a status report (status.wants_report), and
# whether a timer or reminder come due is a job for one of his tools (timers.to_do). Each
# asked only when words suggest it (guards.STATUS_ASKED, guards.ACTION_WORDS), so these are
# all lines with those words in. Made up, after live ones.
STATUS_LINES: list[tuple[str, bool]] = [  # (what they say, asking him for one?)
    ("Status report, please.", True), ("Holly, give me a status report.", True),
    ("Could I have a status update please, Holly?", True), ("No, I need another status report please", True),
    ("What's the status report today?", True), ("Status update?", True), ("Morning Holly, status report.", True),
    ("Go on then, let's have a status report.", True),
    ("I've got a status report to write for work.", False), ("Ugh, my boss wants a status update by five.", False),
    ("That status report was a bit grim.", False), ("The project status report is due on Friday.", False),
    ("I sent the status update to Jo already.", False), ("Do you think my status report is too long?", False),
]
ACTION_LABELS: list[tuple[str, bool]] = [  # (a timer's label, his job with a tool?)
    ("play some Metallica", True), ("check the weather in Keswick", True), ("tell me the time", True),
    ("read me the headlines", True), ("put my liked songs on", True), ("play the album Remain in Light", True),
    ("look up the football scores", True), ("what's the weather doing tomorrow", True),
    ("pasta", False), ("go to Jo's school play", False), ("play football with Sam", False),
    ("search for my keys", False), ("watch the news at six", False), ("find out about the boiler from Jo", False),
    ("take the washing in before the weather turns", False), ("Jo's music lesson", False),
]
# Which tool each timer job needs (timers.job_tool): any of a set where either would do.
ACTION_TOOLS: list[tuple[str, set[str]]] = [
    ("play some Metallica", {"spotify"}), ("check the weather in Keswick", {"weather"}),
    ("tell me the time", {"clock"}), ("read me the headlines", {"news"}), ("put my liked songs on", {"spotify"}),
    ("play the album Remain in Light", {"spotify"}), ("look up the football scores", {"search", "news"}),
    ("what's the weather doing tomorrow", {"weather"}), ("what's in the tech news", {"news"}),
    ("find out what time the tip opens", {"search"}), ("put some Talking Heads on", {"spotify"}),
    ("what time is it in Tokyo", {"clock"}),
]
# What he last said, the crew's reply, and whether they're taking him up on research
# (offers.taken_up). The first is real (25 November): an offer said, not asked.
LINKS = "I can't follow links, I'm a computer, not a browser. I can do a proper research job on it if you want."
GIST = "I can give you the gist of what I found, or I could do a proper research job on it if you're feeling ambitious."
WHICH = "Do you want me to just search for it now, or do a proper report while I'm dozing?"
# ...and whether they're taking him up on something he offered with a tool (offers.taken_up).
# The recipe one is real too (26 November): "You can get it now if you like" had him promise,
# and call nothing.
RECIPE = "If you're really going to bother, I'll find one. Want me to search for it now or later?"
TAKEN_UP_LINES: list[tuple[tuple[str, str], bool]] = [
    ((LINKS, "Yes, please."), True), ((LINKS, "Go on then."), True), ((LINKS, "No, you're all right."), False),
    # (The gist: a tool is fine, the story's article read: app/news.py.)
    ((GIST, "Go on then, do it properly."), True), ((GIST, "Just the gist."), True),
    ((WHICH, "A proper report, please."), True), ((WHICH, "Just search it now."), True),
    ((WHICH, "Actually, never mind."), False),
    ((RECIPE, "You can get it now if you like"), True), ((RECIPE, "Later's fine."), True),
    (("I can search for one if you're actually planning on cooking them.", "Go on then."), True),
    (("The board games report's ready. Want a summary?", "Yes, please."), True),
    (("That's the gist of the report. Want me to open it, or talk you through it?", "Open it."), True),
    (("Want me to set a timer for the pasta?", "Go on, ten minutes."), True),
    (("Did you mean Keswick, in Cumbria?", "Yes, that one."), True),
    (("Want the full article, or is that enough?", "That's enough, thanks."), False),
    (("That's a proper mess, that is.", "Yes, it is."), False),
    (("It's a bit of a cry for help, isn't it?", "Yes, I suppose it is."), False),
    (("Rain. One hundred per cent chance of it. Good luck with that.", "Thanks."), False),
    # Taken for yes in a full run (26 November): their own news, and a new request of theirs.
    (("January. That's a long way off. Do you want a recipe for that lasagne now, or are you just telling me "
      "your life story?", "I start in January. I'm so excited."), False),
    (("About time. Do you want a recipe for that lasagne now, then?", "I start in January. I'm so excited."), False),
    (("Science — A comet is coming this weekend and bees can apparently count. Not much to do, is there?",
      "Can you tell me more about the comet story?"), False),
]


TOOL_NAMES = ("search", "research", "news", "weather", "clock", "timer", "spotify", "recall", "status")
# Which tool each offer taken up needs (offers.offered_tool): (what he said, their reply), and
# the tools that would do. The second, a search taken up, started a report 2 times in 16.
OFFERED_TOOL_LINES: list[tuple[tuple, set[str]]] = [  # ((said, reply[, tools his reply came from]), tools)
    ((LINKS, "Yes, please."), {"research"}), ((LINKS, "Go on then."), {"research"}),
    ((RECIPE, "You can get it now if you like"), {"search"}), ((RECIPE, "Later's fine."), {"search", "research"}),  # no "search later": a report while he dozes will do
    ((GIST, "Go on then, do it properly."), {"research"}), ((GIST, "Just the gist."), {"news", "search"}),
    ((WHICH, "A proper report, please."), {"research"}), ((WHICH, "Just search it now."), {"search"}),
    (("I can search for one if you're actually planning on cooking them.", "Go on then."), {"search"}),
    (("The board games report's ready. Want a summary?", "Yes, please."), {"research"}),
    (("That's the gist of the report. Want me to open it, or talk you through it?", "Open it."), {"research"}),
    (("Want me to set a timer for the pasta?", "Go on, ten minutes."), {"timer"}),
    (("Did you mean Keswick, in Cumbria?", "Yes, that one."), {"weather", "search"}),
    # Held out:
    (("I could put some music on, if you like. Something cheerful?", "Go on then."), {"spotify"}),
    (("Want me to remind you about the bins in the morning?", "Yes please, seven o'clock."), {"timer"}),
    (("I could dig up the latest on that, or write you a proper report on it overnight.", "The report."),
     {"research"}),
    (("Want me to look up when the tip opens?", "Yeah, go on."), {"search"}),
    (("Shall I read you the rest of that story?", "Go on.", ("news",)), {"news"}),  # research, without "news"
]


# Each decision's labelled cases, by its name in app/decisions.py.
CASES = {"about_the_past": MEMORY_LINES, "memory_offered": MEMORY_OFFERS, "wants_report": STATUS_LINES,
         "to_do": ACTION_LABELS, "job_tool": ACTION_TOOLS, "taken_up": TAKEN_UP_LINES,
         "offered_tool": OFFERED_TOOL_LINES}

# The evals' side checks (scenarios.side_checks): its label on the scorecard, the decision, its
# cases, its inputs from a case, and whether LM Studio failing is its fallback (quiet) or an ERROR.
SIDE_CHECKS = [
    ("status report", "wants_report", STATUS_LINES, lambda line: (line,), False),
    ("timer job", "to_do", ACTION_LABELS, lambda label: (label,), False),
    ("a timer job's tool", "job_tool", ACTION_TOOLS, lambda label: (label, TOOL_NAMES), False),
    ("an offer taken up", "taken_up", TAKEN_UP_LINES, lambda pair: pair, False),
    ("which tool it needs", "offered_tool", OFFERED_TOOL_LINES, lambda x: (x[0], x[1], TOOL_NAMES, *x[2:]), True),
]
