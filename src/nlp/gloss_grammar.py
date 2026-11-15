# -*- coding: utf-8 -*-
"""
SignTalk AI: synthetic gloss -> sentence corpus for the sentence Transformer.

INCLUDE has isolated words only, so there are no signed sentences to learn from. This module
writes the training pairs instead: short ISL-style gloss sequences (topic first, no copula or
articles, e.g. "I MOTHER SICK") with the English and Hindi sentence each one stands for
("My mother is sick." / "मेरी माँ बीमार हैं।"). The Transformer learns the patterns, the
browser runs it after the ST-GCN has recognised the signs of a phrase.

Hindi agreement handled here: adjective gender/number (-ा/-ी/-े), possessives (मेरा/मेरी/मेरे),
the copula (है/हैं/हूँ/हो, था/थी/थे/थीं, होगा/...), and the plural verb for respected people.
Please proof-read: these are rule-built sentences, not a linguist's corpus.
"""
import random
from typing import Dict, List, Tuple

Pair = Tuple[Tuple[str, ...], str, str]  # (gloss tokens, English, Hindi)

# --------------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------------

# key: (English, be-present, be-past, possessive, Hindi, gender, plural, Hindi possessive stem)
PRONOUNS = {
    "i": ("I", "am", "was", "my", "मैं", "m", False, "मेर"),
    "you": ("you", "are", "were", "your", "तुम", "m", True, "तुम्हार"),
    "he": ("he", "is", "was", "his", "वह", "m", False, "उसक"),
    "she": ("she", "is", "was", "her", "वह", "f", False, "उसक"),
    "we": ("we", "are", "were", "our", "हम", "m", True, "हमार"),
    "youplural": ("you all", "are", "were", "your", "आप सब", "m", True, "आपक"),
    "they": ("they", "are", "were", "their", "वे", "m", True, "उनक"),
}
HI_COPULA_PRONOUN = {  # present, past, future
    "i": ("हूँ", "था", "होऊँगा"), "you": ("हो", "थे", "होगे"), "he": ("है", "था", "होगा"),
    "she": ("है", "थी", "होगी"), "we": ("हैं", "थे", "होंगे"), "youplural": ("हैं", "थे", "होंगे"),
    "they": ("हैं", "थे", "होंगे"),
}

# People. key: (English, English plural, Hindi, gender, honorific/plural verb, kinship)
PEOPLE = {
    "son": ("son", "sons", "बेटा", "m", False, True),
    "daughter": ("daughter", "daughters", "बेटी", "f", False, True),
    "mother": ("mother", "mothers", "माँ", "f", True, True),
    "father": ("father", "fathers", "पिता", "m", True, True),
    "parent": ("parents", "parents", "माता-पिता", "m", True, True),
    "baby": ("baby", "babies", "शिशु", "m", False, False),
    "man": ("man", "men", "आदमी", "m", False, False),
    "woman": ("woman", "women", "औरत", "f", False, False),
    "brother": ("brother", "brothers", "भाई", "m", False, True),
    "sister": ("sister", "sisters", "बहन", "f", False, True),
    "family": ("family", "families", "परिवार", "m", False, True),
    "grandfather": ("grandfather", "grandfathers", "दादा", "m", True, True),
    "grandmother": ("grandmother", "grandmothers", "दादी", "f", True, True),
    "husband": ("husband", "husbands", "पति", "m", False, True),
    "wife": ("wife", "wives", "पत्नी", "f", False, True),
    "king": ("king", "kings", "राजा", "m", True, False),
    "queen": ("queen", "queens", "रानी", "f", True, False),
    "president": ("president", "presidents", "राष्ट्रपति", "m", True, False),
    "neighbour": ("neighbour", "neighbours", "पड़ोसी", "m", False, True),
    "boy": ("boy", "boys", "लड़का", "m", False, False),
    "girl": ("girl", "girls", "लड़की", "f", False, False),
    "child": ("child", "children", "बच्चा", "m", False, True),
    "adult": ("adult", "adults", "वयस्क", "m", False, False),
    "friend": ("friend", "friends", "दोस्त", "m", False, True),
    "player": ("player", "players", "खिलाड़ी", "m", False, False),
    "teacher": ("teacher", "teachers", "शिक्षक", "m", True, True),
    "student": ("student", "students", "विद्यार्थी", "m", False, False),
    "lawyer": ("lawyer", "lawyers", "वकील", "m", False, True),
    "doctor": ("doctor", "doctors", "डॉक्टर", "m", False, True),
    "patient": ("patient", "patients", "मरीज़", "m", False, False),
    "waiter": ("waiter", "waiters", "वेटर", "m", False, False),
    "secretary": ("secretary", "secretaries", "सचिव", "m", False, True),
    "priest": ("priest", "priests", "पुजारी", "m", True, False),
    "police": ("police officer", "police officers", "पुलिस अधिकारी", "m", False, False),
    "soldier": ("soldier", "soldiers", "सैनिक", "m", False, False),
    "artist": ("artist", "artists", "कलाकार", "m", False, False),
    "author": ("author", "authors", "लेखक", "m", False, False),
    "manager": ("manager", "managers", "प्रबंधक", "m", False, True),
    "reporter": ("reporter", "reporters", "संवाददाता", "m", False, False),
    "actor": ("actor", "actors", "अभिनेता", "m", False, False),
}
# "X is a <job>" nouns; female Hindi forms where Hindi has one
JOBS = ["teacher", "student", "lawyer", "doctor", "patient", "waiter", "secretary", "priest", "police",
        "soldier", "artist", "author", "manager", "reporter", "actor", "player", "friend", "neighbour"]
HI_FEMININE = {"teacher": "शिक्षिका", "author": "लेखिका", "actor": "अभिनेत्री"}

# Person adjectives. key: (English, Hindi base form)
PERSON_ADJ = {
    "happy": ("happy", "खुश"), "sad": ("sad", "उदास"), "beautiful": ("beautiful", "सुंदर"),
    "ugly": ("ugly", "बदसूरत"), "deaf": ("deaf", "बधिर"), "blind": ("blind", "दृष्टिहीन"),
    "rich": ("rich", "अमीर"), "poor": ("poor", "गरीब"), "sick": ("sick", "बीमार"),
    "healthy": ("healthy", "स्वस्थ"), "strong": ("strong", "मज़बूत"), "weak": ("weak", "कमज़ोर"),
    "young": ("young", "जवान"), "old": ("old", "बूढ़ा"), "tall": ("tall", "लंबा"),
    "short": ("short", "नाटा"), "famous": ("famous", "प्रसिद्ध"), "good": ("good", "अच्छा"),
    "bad": ("bad", "बुरा"), "quiet": ("quiet", "शांत"), "fast": ("fast", "तेज़"),
    "slow": ("slow", "धीमा"), "alive": ("alive", "जीवित"), "dead": ("dead", "मृत"),
    "mean": ("mean", "निर्दयी"), "alright": ("fine", "ठीक"), "pleased": ("pleased", "प्रसन्न"),
}

# Thing adjectives. key: (English, Hindi base form)
THING_ADJ = {
    "expensive": ("expensive", "महंगा"), "cheap": ("cheap", "सस्ता"), "new": ("new", "नया"),
    "old": ("old", "पुराना"), "biglarge": ("big", "बड़ा"), "smalllittle": ("small", "छोटा"),
    "heavy": ("heavy", "भारी"), "light": ("light", "हल्का"), "clean": ("clean", "साफ़"),
    "dirty": ("dirty", "गंदा"), "hot": ("hot", "गरम"), "cold": ("cold", "ठंडा"),
    "wet": ("wet", "गीला"), "dry": ("dry", "सूखा"), "soft": ("soft", "मुलायम"),
    "hard": ("hard", "सख़्त"), "long": ("long", "लंबा"), "short": ("short", "छोटा"),
    "wide": ("wide", "चौड़ा"), "narrow": ("narrow", "संकरा"), "thick": ("thick", "मोटा"),
    "flat": ("flat", "समतल"), "curved": ("curved", "घुमावदार"), "tight": ("tight", "तंग"),
    "loose": ("loose", "ढीला"), "beautiful": ("beautiful", "सुंदर"), "ugly": ("ugly", "बदसूरत"),
    "good": ("good", "अच्छा"), "bad": ("bad", "बुरा"), "fast": ("fast", "तेज़"),
    "slow": ("slow", "धीमा"), "famous": ("famous", "प्रसिद्ध"), "quiet": ("quiet", "शांत"),
    "loud": ("loud", "तेज़"), "high": ("high", "ऊँचा"), "low": ("low", "कम"),
}
ANIMAL_ADJ = {"biglarge": ("big", "बड़ा"), "smalllittle": ("small", "छोटा"), "fast": ("fast", "तेज़"),
              "slow": ("slow", "धीमा"), "beautiful": ("beautiful", "सुंदर"), "clean": ("clean", "साफ़"),
              "dirty": ("dirty", "गंदा"), "strong": ("strong", "मज़बूत"), "weak": ("weak", "कमज़ोर"),
              "sick": ("sick", "बीमार"), "healthy": ("healthy", "स्वस्थ"), "old": ("old", "बूढ़ा"),
              "young": ("young", "जवान"), "good": ("good", "अच्छा"), "dead": ("dead", "मृत"),
              "alive": ("alive", "जीवित"), "happy": ("happy", "खुश"), "quiet": ("quiet", "शांत")}

COLOURS = {"red": ("red", "लाल"), "green": ("green", "हरा"), "blue": ("blue", "नीला"),
           "yellow": ("yellow", "पीला"), "brown": ("brown", "भूरा"), "pink": ("pink", "गुलाबी"),
           "orange": ("orange", "नारंगी"), "black": ("black", "काला"), "white": ("white", "सफ़ेद"),
           "grey": ("grey", "स्लेटी")}

# Animals and things. key: (English, English plural verb?, Hindi, gender, Hindi plural verb?)
ANIMALS = {
    "dog": ("dog", False, "कुत्ता", "m", False), "cat": ("cat", False, "बिल्ली", "f", False),
    "fish": ("fish", False, "मछली", "f", False), "bird": ("bird", False, "पक्षी", "m", False),
    "cow": ("cow", False, "गाय", "f", False), "mouse": ("mouse", False, "चूहा", "m", False),
    "horse": ("horse", False, "घोड़ा", "m", False), "animal": ("animal", False, "जानवर", "m", False),
}
CLOTHES = {
    "hat": ("hat", False, "टोपी", "f", False), "dress": ("dress", False, "पोशाक", "f", False),
    "suit": ("suit", False, "सूट", "m", False), "skirt": ("skirt", False, "स्कर्ट", "f", False),
    "shirt": ("shirt", False, "कमीज़", "f", False), "tshirt": ("T-shirt", False, "टी-शर्ट", "f", False),
    "pant": ("pants", True, "पैंट", "f", False), "shoes": ("shoes", True, "जूते", "m", True),
    "clothing": ("clothes", True, "कपड़े", "m", True),
}
ELECTRONICS = {
    "clock": ("clock", False, "घड़ी", "f", False), "lamp": ("lamp", False, "लैंप", "m", False),
    "fan": ("fan", False, "पंखा", "m", False), "cellphone": ("phone", False, "मोबाइल फ़ोन", "m", False),
    "computer": ("computer", False, "कंप्यूटर", "m", False), "laptop": ("laptop", False, "लैपटॉप", "m", False),
    "screen": ("screen", False, "स्क्रीन", "f", False), "camera": ("camera", False, "कैमरा", "m", False),
    "television": ("television", False, "टेलीविज़न", "m", False), "radio": ("radio", False, "रेडियो", "m", False),
    "telephone": ("telephone", False, "टेलीफ़ोन", "m", False),
}
HOME = {
    "table": ("table", False, "मेज़", "f", False), "chair": ("chair", False, "कुर्सी", "f", False),
    "bed": ("bed", False, "बिस्तर", "m", False), "window": ("window", False, "खिड़की", "f", False),
    "door": ("door", False, "दरवाज़ा", "m", False), "pencil": ("pencil", False, "पेंसिल", "f", False),
    "pen": ("pen", False, "पेन", "m", False), "photograph": ("photo", False, "तस्वीर", "f", False),
    "soap": ("soap", False, "साबुन", "m", False), "book": ("book", False, "किताब", "f", False),
    "page": ("page", False, "पन्ना", "m", False), "key": ("key", False, "चाबी", "f", False),
    "letter": ("letter", False, "पत्र", "m", False), "paper": ("paper", False, "कागज़", "m", False),
    "lock": ("lock", False, "ताला", "m", False), "bag": ("bag", False, "थैला", "m", False),
    "box": ("box", False, "डिब्बा", "m", False), "gift": ("gift", False, "उपहार", "m", False),
    "card": ("card", False, "कार्ड", "m", False), "ring": ("ring", False, "अंगूठी", "f", False),
    "tool": ("tool", False, "औज़ार", "m", False), "ball": ("ball", False, "गेंद", "f", False),
    "newspaper": ("newspaper", False, "अख़बार", "m", False), "medicine": ("medicine", False, "दवा", "f", False),
    "money": ("money", False, "पैसे", "m", True), "gun": ("gun", False, "बंदूक", "f", False),
    "bill": ("bill", False, "बिल", "m", False),
}
VEHICLES = {
    "train": ("train", False, "रेलगाड़ी", "f", False), "plane": ("plane", False, "हवाई जहाज़", "m", False),
    "car": ("car", False, "गाड़ी", "f", False), "truck": ("truck", False, "ट्रक", "m", False),
    "bicycle": ("bicycle", False, "साइकिल", "f", False), "bus": ("bus", False, "बस", "f", False),
    "boat": ("boat", False, "नाव", "f", False), "trainticket": ("train ticket", False, "रेल टिकट", "m", False),
}
THINGS = {**CLOTHES, **ELECTRONICS, **HOME, **VEHICLES}

# Which adjectives / colours make sense for which group
GROUP_ADJ = {
    "clothes": ["new", "old", "clean", "dirty", "expensive", "cheap", "beautiful", "tight", "loose",
                "long", "short", "wet", "dry", "soft", "good", "bad"],
    "electronics": ["new", "old", "expensive", "cheap", "biglarge", "smalllittle", "heavy", "light",
                    "good", "bad", "loud", "quiet", "fast", "slow"],
    "home": ["new", "old", "biglarge", "smalllittle", "heavy", "light", "clean", "dirty", "expensive",
             "cheap", "good", "bad", "beautiful", "soft", "hard", "wet", "dry", "thick", "flat", "long", "short"],
    "vehicles": ["new", "old", "biglarge", "smalllittle", "fast", "slow", "expensive", "cheap", "clean",
                 "dirty", "beautiful", "good", "bad"],
}
NO_ADJ = {"money", "bill", "medicine", "newspaper", "letter", "gun", "trainticket"}
COLOURED = set(CLOTHES) | {"clock", "lamp", "fan", "cellphone", "computer", "laptop", "camera", "table",
                           "chair", "bed", "door", "window", "pencil", "pen", "book", "bag", "box", "card",
                           "ball", "car", "bus", "bicycle", "boat", "truck", "train", "plane"}
ANIMAL_COLOURS = ["white", "black", "brown", "grey"]

# Places. key: (English subject, English location, Hindi, gender, Hindi location)
PLACES = {
    "house": ("the house", "at home", "घर", "m", "घर पर"),
    "bedroom": ("the bedroom", "in the bedroom", "शयनकक्ष", "m", "शयनकक्ष में"),
    "kitchen": ("the kitchen", "in the kitchen", "रसोई", "f", "रसोई में"),
    "bathroom": ("the bathroom", "in the bathroom", "स्नानघर", "m", "स्नानघर में"),
    "city": ("the city", "in the city", "शहर", "m", "शहर में"),
    "streetorroad": ("the road", "on the road", "सड़क", "f", "सड़क पर"),
    "trainstation": ("the train station", "at the train station", "रेलवे स्टेशन", "m", "रेलवे स्टेशन पर"),
    "restaurant": ("the restaurant", "at the restaurant", "रेस्टोरेंट", "m", "रेस्टोरेंट में"),
    "court": ("the court", "at the court", "अदालत", "f", "अदालत में"),
    "school": ("the school", "at school", "स्कूल", "m", "स्कूल में"),
    "office": ("the office", "at the office", "दफ़्तर", "m", "दफ़्तर में"),
    "university": ("the university", "at the university", "विश्वविद्यालय", "m", "विश्वविद्यालय में"),
    "park": ("the park", "in the park", "पार्क", "m", "पार्क में"),
    "storeorshop": ("the shop", "at the shop", "दुकान", "f", "दुकान पर"),
    "library": ("the library", "in the library", "पुस्तकालय", "m", "पुस्तकालय में"),
    "hospital": ("the hospital", "in the hospital", "अस्पताल", "m", "अस्पताल में"),
    "temple": ("the temple", "at the temple", "मंदिर", "m", "मंदिर में"),
    "market": ("the market", "at the market", "बाज़ार", "m", "बाज़ार में"),
    "india": ("India", "in India", "भारत", "m", "भारत में"),
    "ground": ("the ground", "on the ground", "मैदान", "m", "मैदान में"),
    "bank": ("the bank", "at the bank", "बैंक", "m", "बैंक में"),
}
PLACE_ADJ = ["biglarge", "smalllittle", "clean", "dirty", "beautiful", "new", "old", "famous", "quiet",
             "good", "bad"]

# Where small things lie. key: (English, Hindi)
CONTAINERS = {
    "table": ("on the table", "मेज़ पर"), "chair": ("on the chair", "कुर्सी पर"),
    "bed": ("on the bed", "बिस्तर पर"), "box": ("in the box", "डिब्बे में"),
    "bag": ("in the bag", "थैले में"), "pocket": ("in the pocket", "जेब में"),
    "car": ("in the car", "गाड़ी में"),
}
SMALL_THINGS = ["book", "pen", "pencil", "key", "cellphone", "money", "paper", "letter", "card", "ring",
                "ball", "gift", "soap", "photograph", "medicine", "clock", "newspaper", "laptop", "camera",
                "tool", "bill", "trainticket", "shoes", "hat", "shirt", "gun"]

# Abstract nouns. key: (English subject, Hindi, gender, Hindi plural verb)
ABSTRACT = {
    "religion": ("religion", "धर्म", "m", False), "marriage": ("the marriage", "शादी", "f", False),
    "team": ("the team", "टीम", "f", False), "technology": ("technology", "तकनीक", "f", False),
    "energy": ("energy", "ऊर्जा", "f", False), "war": ("war", "युद्ध", "m", False),
    "peace": ("peace", "शांति", "f", False), "election": ("the election", "चुनाव", "m", False),
    "sport": ("sport", "खेल", "m", False), "exercise": ("exercise", "व्यायाम", "m", False),
    "science": ("science", "विज्ञान", "m", False), "job": ("the job", "नौकरी", "f", False),
    "dream": ("the dream", "सपना", "m", False), "god": ("God", "भगवान", "m", True),
    "crowd": ("the crowd", "भीड़", "f", False), "time": ("the time", "समय", "m", False),
    "colour": ("the colour", "रंग", "m", False), "season": ("the season", "मौसम", "m", False),
    "death": ("death", "मृत्यु", "f", False), "attack": ("the attack", "हमला", "m", False),
}
ABSTRACT_ADJ = {"good", "bad"}
SIZE_NOUNS = {"crowd", "team"}  # also big / small

DAYS = {"sunday": ("Sunday", "रविवार"), "monday": ("Monday", "सोमवार"), "tuesday": ("Tuesday", "मंगलवार"),
        "wednesday": ("Wednesday", "बुधवार"), "thursday": ("Thursday", "गुरुवार"),
        "friday": ("Friday", "शुक्रवार"), "saturday": ("Saturday", "शनिवार")}
TENSE_WORDS = {"today": ("Today", "आज", 0), "yesterday": ("Yesterday", "कल", 1), "tomorrow": ("Tomorrow", "कल", 2)}
DAY_PARTS = {"morning": ("In the morning", "सुबह"), "afternoon": ("In the afternoon", "दोपहर में"),
             "evening": ("In the evening", "शाम को"), "night": ("At night", "रात को")}
SEASONS = {"summer": ("Summer", "गर्मी"), "winter": ("Winter", "सर्दी"), "spring": ("Spring", "वसंत"),
           "fall": ("Fall", "पतझड़"), "monsoon": ("The monsoon", "मानसून")}
SEASON_ADJ = {"hot": ("hot", "गरम"), "cold": ("cold", "ठंडा"), "beautiful": ("beautiful", "सुंदर"),
              "wet": ("wet", "गीला"), "dry": ("dry", "सूखा"), "good": ("good", "अच्छा"),
              "warm": ("warm", "हल्का गरम"), "cool": ("cool", "ठंडा")}
WEATHER = {"hot": ("hot", "गरमी", "f"), "cold": ("cold", "ठंड", "f")}

GREETINGS = {"hello": ("Hello", "नमस्ते", "!"), "goodmorning": ("Good morning", "शुभ प्रभात", "!"),
             "goodafternoon": ("Good afternoon", "शुभ दोपहर", "!"),
             "goodevening": ("Good evening", "शुभ संध्या", "!"), "goodnight": ("Good night", "शुभ रात्रि", "!"),
             "thankyou": ("Thank you", "धन्यवाद", ".")}
ADDRESSABLE = ["teacher", "friend", "mother", "father", "brother", "sister", "doctor", "grandfather",
               "grandmother", "neighbour", "son", "daughter", "manager", "student", "wife", "husband"]

FIXED: List[Pair] = [
    (("howareyou",), "How are you?", "आप कैसे हैं?"),
    (("alright",), "All right.", "ठीक है।"),
    (("pleased",), "Pleased to meet you.", "आपसे मिलकर खुशी हुई।"),
    (("hello", "howareyou"), "Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
    (("hello", "pleased"), "Hello, pleased to meet you.", "नमस्ते, आपसे मिलकर खुशी हुई।"),
    (("i", "alright", "thankyou"), "I am fine, thank you.", "मैं ठीक हूँ, धन्यवाद।"),
    (("time", "house"), "Time to go home.", "घर जाने का समय है।"),
]

# --------------------------------------------------------------------------------------------
# Inflection helpers
# --------------------------------------------------------------------------------------------

def hi_adj(base: str, gender: str, plural: bool) -> str:
    """Hindi adjective agreement: अच्छा / अच्छी / अच्छे (नया / नई / नए); others don't change."""
    if base == "नया":
        return "नई" if gender == "f" else ("नए" if plural else "नया")
    if base.endswith("ा"):
        if gender == "f":
            return base[:-1] + "ी"
        return base[:-1] + "े" if plural else base
    return base


def hi_poss(stem: str, gender: str, plural: bool) -> str:
    """मेरा / मेरी / मेरे for the possessed noun's gender and number."""
    return stem + ("ी" if gender == "f" else ("े" if plural else "ा"))


def hi_copula(gender: str, plural: bool, tense: int = 0) -> str:
    """है / हैं, था / थी / थे / थीं, होगा / होगी / होंगे / होंगी for a noun subject."""
    if tense == 0:
        return "हैं" if plural else "है"
    if tense == 1:
        return {("m", False): "था", ("f", False): "थी", ("m", True): "थे", ("f", True): "थीं"}[(gender, plural)]
    return {("m", False): "होगा", ("f", False): "होगी", ("m", True): "होंगे", ("f", True): "होंगी"}[(gender, plural)]


def en_be(plural: bool, tense: int = 0) -> str:
    return [("is", "are"), ("was", "were"), ("will be", "will be")][tense][int(plural)]


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeio" else "a"


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def en_subject(key: str, poss: str = None) -> str:
    """'my mother', 'Mother', 'the boy', 'my car', 'the car'."""
    if key in PEOPLE:
        en, kin = PEOPLE[key][0], PEOPLE[key][5]
        if poss:
            return f"{poss} {en}"
        return cap(en) if kin and key not in ("family", "friend", "neighbour", "child", "lawyer", "doctor",
                                               "secretary", "manager", "parent") else f"the {en}"
    en = (ANIMALS.get(key) or THINGS[key])[0]
    return f"{poss} {en}" if poss else f"the {en}"


def noun_info(key: str):
    """(English plural verb, Hindi, gender, Hindi plural verb) for a person / animal / thing."""
    if key in PEOPLE:
        en, _, hi, g, hon, _ = PEOPLE[key]
        return key == "parent", hi, g, hon
    en, en_pl, hi, g, hi_pl = (ANIMALS.get(key) or THINGS[key])
    return en_pl, hi, g, hi_pl


def sentence(en: str, hi: str, end_en: str = ".", end_hi: str = "।") -> Tuple[str, str]:
    return cap(en) + end_en, hi + end_hi


# --------------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------------

def p_pron_adj(r: random.Random) -> Pair:
    p = r.choice(list(PRONOUNS)); a = r.choice(list(PERSON_ADJ))
    en, be, _, _, hi, g, pl, _ = PRONOUNS[p]
    return (p, a), *sentence(f"{en} {be} {PERSON_ADJ[a][0]}", f"{hi} {hi_adj(PERSON_ADJ[a][1], g, pl)} {HI_COPULA_PRONOUN[p][0]}")


def p_tense_pron_adj(r: random.Random) -> Pair:
    t = r.choice(list(TENSE_WORDS)); p = r.choice(list(PRONOUNS)); a = r.choice(list(PERSON_ADJ))
    t_en, t_hi, tense = TENSE_WORDS[t]
    en, be, was, _, hi, g, pl, _ = PRONOUNS[p]
    verb = [be, was, "will be"][tense]
    subj = en if p != "i" else "I"
    return (t, p, a), *sentence(f"{t_en} {subj} {verb} {PERSON_ADJ[a][0]}",
                                f"{t_hi} {hi} {hi_adj(PERSON_ADJ[a][1], g, pl)} {HI_COPULA_PRONOUN[p][tense]}")


def _person_subject(r: random.Random, with_poss: bool):
    n = r.choice([k for k in PEOPLE if PEOPLE[k][5]] if with_poss else list(PEOPLE))
    en_pl, hi, g, hon = noun_info(n)
    if with_poss:
        p = r.choice([q for q in PRONOUNS if (q, n) not in (("he", "husband"), ("she", "wife"))])
        P = PRONOUNS[p]
        return (p, n), en_subject(n, P[3]), f"{hi_poss(P[7], g, hon)} {hi}", g, hon, en_pl
    return (n,), en_subject(n), hi, g, hon, en_pl


def p_person_adj(r: random.Random) -> Pair:
    toks, en_s, hi_s, g, hon, en_pl = _person_subject(r, r.random() < 0.5)
    a = r.choice([k for k in PERSON_ADJ if k not in ("alright", "pleased")])
    return (*toks, a), *sentence(f"{en_s} {en_be(en_pl)} {PERSON_ADJ[a][0]}",
                                 f"{hi_s} {hi_adj(PERSON_ADJ[a][1], g, hon)} {hi_copula(g, hon)}")


def p_pron_job(r: random.Random) -> Pair:
    p = r.choice(list(PRONOUNS)); j = r.choice(JOBS)
    en, be, _, _, hi, g, pl, _ = PRONOUNS[p]
    en_plural = p in ("we", "youplural", "they")  # Hindi तुम takes plural agreement, English "you" here is one person
    job_en = PEOPLE[j][1] if en_plural else f"{article(PEOPLE[j][0])} {PEOPLE[j][0]}"
    job_hi = HI_FEMININE.get(j, PEOPLE[j][2]) if g == "f" else PEOPLE[j][2]
    return (p, j), *sentence(f"{en} {be} {job_en}", f"{hi} {job_hi} {HI_COPULA_PRONOUN[p][0]}")


def p_person_job(r: random.Random) -> Pair:
    toks, en_s, hi_s, g, hon, en_pl = _person_subject(r, r.random() < 0.7)
    j = r.choice([x for x in JOBS if x != toks[-1]])
    job_en = PEOPLE[j][1] if en_pl else f"{article(PEOPLE[j][0])} {PEOPLE[j][0]}"
    job_hi = HI_FEMININE.get(j, PEOPLE[j][2]) if g == "f" else PEOPLE[j][2]
    return (*toks, j), *sentence(f"{en_s} {en_be(en_pl)} {job_en}", f"{hi_s} {job_hi} {hi_copula(g, hon)}")


def p_pron_place(r: random.Random) -> Pair:
    p = r.choice(list(PRONOUNS)); pl_ = r.choice(list(PLACES))
    en, be, _, _, hi, _, _, _ = PRONOUNS[p]
    return (p, pl_), *sentence(f"{en} {be} {PLACES[pl_][1]}", f"{hi} {PLACES[pl_][4]} {HI_COPULA_PRONOUN[p][0]}")


def p_person_place(r: random.Random) -> Pair:
    toks, en_s, hi_s, g, hon, en_pl = _person_subject(r, r.random() < 0.5)
    pl_ = r.choice(list(PLACES))
    return (*toks, pl_), *sentence(f"{en_s} {en_be(en_pl)} {PLACES[pl_][1]}", f"{hi_s} {PLACES[pl_][4]} {hi_copula(g, hon)}")


def p_time_pron_place(r: random.Random) -> Pair:
    p = r.choice(list(PRONOUNS)); pl_ = r.choice(list(PLACES))
    en, be, was, _, hi, _, _, _ = PRONOUNS[p]
    subj = en if p != "i" else "I"
    if r.random() < 0.5:
        t = r.choice(list(DAY_PARTS)); t_en, t_hi = DAY_PARTS[t]; tense = 0
        when_en, when_hi = t_en, t_hi
    elif r.random() < 0.5:
        t = r.choice(list(DAYS)); when_en, when_hi = f"On {DAYS[t][0]}", f"{DAYS[t][1]} को"; tense = 0
    else:
        t = r.choice(list(TENSE_WORDS)); when_en, when_hi, tense = TENSE_WORDS[t]
    verb = [be, was, "will be"][tense]
    return (t, p, pl_), *sentence(f"{when_en} {subj} {verb} {PLACES[pl_][1]}",
                                  f"{when_hi} {hi} {PLACES[pl_][4]} {HI_COPULA_PRONOUN[p][tense]}")


def _thing(r: random.Random, pool) -> str:
    return r.choice([k for k in pool if k not in NO_ADJ])


def _group(key: str) -> str:
    for name, d in (("clothes", CLOTHES), ("electronics", ELECTRONICS), ("home", HOME), ("vehicles", VEHICLES)):
        if key in d:
            return name
    raise KeyError(key)


def p_thing_adj(r: random.Random) -> Pair:
    t = _thing(r, THINGS); a = r.choice(GROUP_ADJ[_group(t)])
    en_pl, hi, g, hi_pl = noun_info(t)
    poss = None
    toks = (t, a)
    hi_s = hi
    if r.random() < 0.5:
        p = r.choice(list(PRONOUNS)); P = PRONOUNS[p]; poss = P[3]
        toks = (p, t, a); hi_s = f"{hi_poss(P[7], g, hi_pl)} {hi}"
    return toks, *sentence(f"{en_subject(t, poss)} {en_be(en_pl)} {THING_ADJ[a][0]}",
                           f"{hi_s} {hi_adj(THING_ADJ[a][1], g, hi_pl)} {hi_copula(g, hi_pl)}")


def p_it_adj(r: random.Random) -> Pair:
    a = r.choice(sorted(set(GROUP_ADJ["home"]) | set(GROUP_ADJ["vehicles"])))
    return ("it", a), *sentence(f"it is {THING_ADJ[a][0]}", f"यह {hi_adj(THING_ADJ[a][1], 'm', False)} है")


def p_thing_colour(r: random.Random) -> Pair:
    t = r.choice(sorted(COLOURED)); c = r.choice(list(COLOURS))
    en_pl, hi, g, hi_pl = noun_info(t)
    poss = None; toks = (t, c); hi_s = hi
    if r.random() < 0.5:
        p = r.choice(list(PRONOUNS)); P = PRONOUNS[p]; poss = P[3]
        toks = (p, t, c); hi_s = f"{hi_poss(P[7], g, hi_pl)} {hi}"
    return toks, *sentence(f"{en_subject(t, poss)} {en_be(en_pl)} {COLOURS[c][0]}",
                           f"{hi_s} {hi_adj(COLOURS[c][1], g, hi_pl)} {hi_copula(g, hi_pl)}")


def p_animal(r: random.Random) -> Pair:
    n = r.choice(list(ANIMALS))
    en_pl, hi, g, hi_pl = noun_info(n)
    if r.random() < 0.7:
        a = r.choice(list(ANIMAL_ADJ)); a_en, a_hi = ANIMAL_ADJ[a]
    else:
        a = r.choice(ANIMAL_COLOURS); a_en, a_hi = COLOURS[a]
    poss = None; toks = (n, a); hi_s = hi
    if r.random() < 0.5:
        p = r.choice(list(PRONOUNS)); P = PRONOUNS[p]; poss = P[3]
        toks = (p, n, a); hi_s = f"{hi_poss(P[7], g, hi_pl)} {hi}"
    return toks, *sentence(f"{en_subject(n, poss)} {en_be(en_pl)} {a_en}",
                           f"{hi_s} {hi_adj(a_hi, g, hi_pl)} {hi_copula(g, hi_pl)}")


def p_thing_in(r: random.Random) -> Pair:
    t = r.choice(SMALL_THINGS); c = r.choice([k for k in CONTAINERS if k != t])
    en_pl, hi, g, hi_pl = noun_info(t)
    poss = None; toks = (t, c); hi_s = hi
    if r.random() < 0.4:
        p = r.choice(list(PRONOUNS)); P = PRONOUNS[p]; poss = P[3]
        toks = (p, t, c); hi_s = f"{hi_poss(P[7], g, hi_pl)} {hi}"
    return toks, *sentence(f"{en_subject(t, poss)} {en_be(en_pl)} {CONTAINERS[c][0]}",
                           f"{hi_s} {CONTAINERS[c][1]} {hi_copula(g, hi_pl)}")


def p_thing_at_place(r: random.Random) -> Pair:
    t = r.choice(SMALL_THINGS + list(VEHICLES)); pl_ = r.choice(list(PLACES))
    en_pl, hi, g, hi_pl = noun_info(t)
    return (t, pl_), *sentence(f"{en_subject(t)} {en_be(en_pl)} {PLACES[pl_][1]}", f"{hi} {PLACES[pl_][4]} {hi_copula(g, hi_pl)}")


def p_place_adj(r: random.Random) -> Pair:
    pl_ = r.choice(list(PLACES))
    pool = PLACE_ADJ + (["wide", "narrow", "long", "flat", "curved"] if pl_ == "streetorroad" else [])
    a = r.choice(pool)
    en_s, _, hi, g, _ = PLACES[pl_]
    return (pl_, a), *sentence(f"{en_s} is {THING_ADJ[a][0]}", f"{hi} {hi_adj(THING_ADJ[a][1], g, False)} है")


def p_abstract(r: random.Random) -> Pair:
    n = r.choice(list(ABSTRACT))
    pool = sorted(ABSTRACT_ADJ | ({"biglarge", "smalllittle"} if n in SIZE_NOUNS else set()))
    a = r.choice(pool)
    en_s, hi, g, hon = ABSTRACT[n]
    return (n, a), *sentence(f"{en_s} is {THING_ADJ[a][0]}", f"{hi} {hi_adj(THING_ADJ[a][1], g, hon)} {hi_copula(g, hon)}")


def p_price(r: random.Random) -> Pair:
    a = r.choice(["high", "low"])
    if r.random() < 0.5:
        t = _thing(r, THINGS)
        en_pl, hi, g, hi_pl = noun_info(t)
        en_t = THINGS[t][0]
        return (t, "price", a), *sentence(f"the price of the {en_t} is {a}",
                                          f"{hi} की क़ीमत {hi_adj(THING_ADJ[a][1], 'f', False)} है")
    return ("price", a), *sentence(f"the price is {a}", f"क़ीमत {hi_adj(THING_ADJ[a][1], 'f', False)} है")


def p_day(r: random.Random) -> Pair:
    t = r.choice(list(TENSE_WORDS)); d = r.choice(list(DAYS))
    t_en, t_hi, tense = TENSE_WORDS[t]
    verb_en = ["is", "was", "is"][tense]
    verb_hi = ["है", "था", "है"][tense]
    return (t, d), *sentence(f"{t_en} {verb_en} {DAYS[d][0]}", f"{t_hi} {DAYS[d][1]} {verb_hi}")


def p_weather(r: random.Random) -> Pair:
    if r.random() < 0.5:
        s = r.choice(list(SEASONS)); a = r.choice(list(SEASON_ADJ))
        s_en, s_hi = SEASONS[s]
        return (s, a), *sentence(f"{s_en} is {SEASON_ADJ[a][0]}", f"{s_hi} का मौसम {SEASON_ADJ[a][1]} होता है")
    t = r.choice(list(TENSE_WORDS)); w = r.choice(list(WEATHER))
    t_en, t_hi, tense = TENSE_WORDS[t]
    en_w, hi_w, g = WEATHER[w]
    en = [f"it is {en_w} today", f"it was {en_w} yesterday", f"it will be {en_w} tomorrow"][tense]
    return (t, w), *sentence(en, f"{t_hi} {hi_w} {hi_copula(g, False, tense)}")


def p_greet(r: random.Random) -> Pair:
    g = r.choice(list(GREETINGS)); n = r.choice(ADDRESSABLE)
    g_en, g_hi, end = GREETINGS[g]
    who_en, who_hi = PEOPLE[n][0], PEOPLE[n][2]
    if r.random() < 0.3:
        return (g, n, "howareyou"), f"{g_en}, {who_en}, how are you?", f"{g_hi} {who_hi}, आप कैसे हैं?"
    return (g, n), f"{g_en}, {who_en}{end}", f"{g_hi}, {who_hi}{end if end == '!' else '।'}"


def p_wear(r: random.Random) -> Pair:
    """INCLUDE has no WEAR sign, so a person followed by clothing means wearing it: "I SHIRT",
    "MOTHER RED DRESS". A colour before the garment describes what is worn; a colour after it
    stays the "My shirt is blue." pattern of p_thing_colour."""
    c = r.choice(list(CLOTHES))
    c_en, c_en_pl, c_hi, c_g, c_hi_pl = CLOTHES[c]
    col = r.choice(list(COLOURS)) if r.random() < 0.5 else None
    garment_en = f"{COLOURS[col][0]} {c_en}" if col else c_en
    if not c_en_pl:
        garment_en = f"{article(garment_en)} {garment_en}"
    garment_hi = f"{hi_adj(COLOURS[col][1], c_g, c_hi_pl)} {c_hi}" if col else c_hi
    if r.random() < 0.5:
        p = r.choice(list(PRONOUNS))
        en, be, _, _, hi, g, pl, _ = PRONOUNS[p]
        toks, en_s, hi_s, aux = (p,), f"{en} {be}", hi, HI_COPULA_PRONOUN[p][0]
    else:
        toks, en_s, hi_s, g, pl, en_pl = _person_subject(r, r.random() < 0.4)
        en_s, aux = f"{en_s} {en_be(en_pl)}", hi_copula(g, pl)
    rah = "रही" if g == "f" else ("रहे" if pl else "रहा")
    return (*toks, *((col,) if col else ()), c), *sentence(f"{en_s} wearing {garment_en}",
                                                          f"{hi_s} {garment_hi} पहन {rah} {aux}")


def clothing_pairs() -> List[Pair]:
    """Every pronoun + garment ("I SHIRT"), pronoun + colour + garment ("I BLUE SHIRT", wearing) and
    pronoun + garment + colour ("I SHIRT BLUE", "My shirt is blue."). Only the sign order tells the
    last two apart, so the model sees all of them instead of a random sample."""
    out = []
    for p, (en, be, _, poss, hi, g, pl, stem) in PRONOUNS.items():
        rah = "रही" if g == "f" else ("रहे" if pl else "रहा")
        aux = HI_COPULA_PRONOUN[p][0]
        for c, (c_en, c_en_pl, c_hi, c_g, c_hi_pl) in CLOTHES.items():
            out.append(((p, c), *sentence(f"{en} {be} wearing {c_en if c_en_pl else article(c_en) + ' ' + c_en}",
                                          f"{hi} {c_hi} पहन {rah} {aux}")))
            for col, (col_en, col_hi) in COLOURS.items():
                worn = f"{col_en} {c_en}" if c_en_pl else f"{article(col_en)} {col_en} {c_en}"
                out.append(((p, col, c), *sentence(f"{en} {be} wearing {worn}",
                                                   f"{hi} {hi_adj(col_hi, c_g, c_hi_pl)} {c_hi} पहन {rah} {aux}")))
                out.append(((p, c, col), *sentence(f"{poss} {c_en} {en_be(c_en_pl)} {col_en}",
                                                   f"{hi_poss(stem, c_g, c_hi_pl)} {c_hi} {hi_adj(col_hi, c_g, c_hi_pl)} {hi_copula(c_g, c_hi_pl)}")))
    return out


def p_single(r: random.Random, vocab: List[str], hindi: Dict[str, str], english: Dict[str, str]) -> Pair:
    w = r.choice(vocab)
    if w in GREETINGS:
        g_en, g_hi, end = GREETINGS[w]
        return (w,), g_en + end, g_hi + (end if end == "!" else "।")
    return (w,), cap(english[w]) + ".", hindi.get(w, english[w]) + "।"


PATTERNS = [  # (pattern, weight)
    (p_pron_adj, 6), (p_tense_pron_adj, 6), (p_person_adj, 8), (p_pron_job, 5), (p_person_job, 6),
    (p_pron_place, 5), (p_person_place, 6), (p_time_pron_place, 6), (p_thing_adj, 10), (p_thing_colour, 6),
    (p_animal, 5), (p_thing_in, 5), (p_thing_at_place, 3), (p_place_adj, 3), (p_abstract, 2), (p_price, 1),
    (p_day, 1), (p_weather, 1), (p_greet, 3), (p_it_adj, 1), (p_wear, 6),
]


def build_corpus(vocab: List[str], english: Dict[str, str], hindi: Dict[str, str],
                 n: int = 60000, seed: int = 0) -> List[Pair]:
    """Unique (gloss, English, Hindi) pairs: every vocabulary word alone, the fixed phrases, and
    `n` sampled pattern sentences. Glosses outside `vocab` are never produced."""
    r = random.Random(seed)
    vocab_set = set(vocab)
    out: Dict[Tuple[str, ...], Pair] = {}
    for w in vocab:
        out[(w,)] = p_single(random.Random(w), [w], hindi, english)
    for pair in FIXED:
        out[pair[0]] = pair
    for pair in clothing_pairs():
        out[pair[0]] = pair
    for g, (g_en, g_hi, end) in GREETINGS.items():  # every "Hello, teacher!" combination
        for who in ADDRESSABLE:
            out[(g, who)] = ((g, who), f"{g_en}, {PEOPLE[who][0]}{end}",
                             f"{g_hi}, {PEOPLE[who][2]}{end if end == '!' else '।'}")
    fns, weights = zip(*PATTERNS)
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        toks, en, hi = r.choices(fns, weights)[0](r)
        if all(t in vocab_set for t in toks) and toks not in out:
            out[toks] = (toks, en, hi)
    return list(out.values())
