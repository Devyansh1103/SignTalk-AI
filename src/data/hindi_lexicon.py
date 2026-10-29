# -*- coding: utf-8 -*-
"""
Hindi (Devanagari) display text for INCLUDE vocabulary labels.
Key = normalized label (lowercase letters only, e.g. 'howareyou', 'tshirt').
Labels missing here fall back to the English translation, so nothing breaks.
Please proof-read these - they are plain dictionary-style translations.
"""

HINDI_LEXICON = {
    # Adjectives
    "loud": "तेज़ आवाज़", "quiet": "शांत", "happy": "खुश", "sad": "उदास",
    "beautiful": "सुंदर", "ugly": "बदसूरत", "deaf": "बधिर", "blind": "दृष्टिहीन",
    "nice": "अच्छा", "mean": "निर्दयी", "rich": "अमीर", "poor": "गरीब",
    "thick": "मोटा", "thin": "पतला", "expensive": "महंगा", "cheap": "सस्ता",
    "flat": "समतल", "curved": "घुमावदार", "male": "पुरुष", "female": "महिला",
    "tight": "तंग", "loose": "ढीला", "good": "अच्छा",
    # Animals
    "dog": "कुत्ता", "cat": "बिल्ली", "fish": "मछली", "bird": "पक्षी",
    "cow": "गाय", "mouse": "चूहा", "horse": "घोड़ा", "animal": "जानवर",
    # Clothes
    "hat": "टोपी", "dress": "पोशाक", "suit": "सूट", "skirt": "स्कर्ट",
    "shirt": "कमीज़", "tshirt": "टी-शर्ट", "pant": "पैंट", "shoes": "जूते",
    "pocket": "जेब", "clothing": "कपड़े",
    # Colours
    "red": "लाल", "green": "हरा", "blue": "नीला", "yellow": "पीला",
    "brown": "भूरा", "pink": "गुलाबी", "orange": "नारंगी", "black": "काला",
    "white": "सफ़ेद", "grey": "स्लेटी", "gray": "स्लेटी", "colour": "रंग",
    # Days and time
    "sunday": "रविवार", "monday": "सोमवार", "tuesday": "मंगलवार",
    "wednesday": "बुधवार", "thursday": "गुरुवार", "friday": "शुक्रवार",
    "saturday": "शनिवार", "today": "आज", "tomorrow": "आने वाला कल",
    "yesterday": "बीता हुआ कल", "week": "सप्ताह", "month": "महीना",
    "year": "साल", "hour": "घंटा", "minute": "मिनट", "second": "सेकंड",
    "morning": "सुबह", "afternoon": "दोपहर", "evening": "शाम", "night": "रात",
    "time": "समय",
    # Electronics
    "clock": "घड़ी", "lamp": "लैंप", "fan": "पंखा", "cellphone": "मोबाइल फ़ोन",
    "computer": "कंप्यूटर", "laptop": "लैपटॉप", "screen": "स्क्रीन",
    "camera": "कैमरा", "television": "टेलीविज़न", "radio": "रेडियो",
    # Greetings
    "hello": "नमस्ते", "howareyou": "आप कैसे हैं", "alright": "ठीक है",
    "goodmorning": "शुभ प्रभात", "goodafternoon": "शुभ दोपहर",
    "goodevening": "शुभ संध्या", "goodnight": "शुभ रात्रि",
    "thankyou": "धन्यवाद", "pleased": "प्रसन्न",
    # Jobs
    "teacher": "शिक्षक", "student": "विद्यार्थी", "lawyer": "वकील",
    "doctor": "डॉक्टर", "patient": "मरीज़", "waiter": "वेटर",
    "secretary": "सचिव", "priest": "पुजारी", "police": "पुलिस",
    "soldier": "सैनिक", "artist": "कलाकार", "author": "लेखक",
    "manager": "प्रबंधक", "reporter": "संवाददाता", "actor": "अभिनेता",
    "job": "नौकरी",
    # Transportation
    "train": "रेलगाड़ी", "plane": "हवाई जहाज़", "car": "गाड़ी", "truck": "ट्रक",
    "bicycle": "साइकिल", "bus": "बस", "boat": "नाव", "trainticket": "रेल टिकट",
    "transportation": "परिवहन",
    # Home
    "table": "मेज़", "chair": "कुर्सी", "bed": "बिस्तर", "dream": "सपना",
    "window": "खिड़की", "door": "दरवाज़ा", "bedroom": "शयनकक्ष", "kitchen": "रसोई",
    "bathroom": "स्नानघर", "pencil": "पेंसिल", "pen": "पेन", "photograph": "तस्वीर",
    "soap": "साबुन", "book": "किताब", "page": "पन्ना", "key": "चाबी",
    "paint": "रंग", "letter": "पत्र", "paper": "कागज़", "lock": "ताला",
    "telephone": "टेलीफ़ोन", "bag": "थैला", "box": "डिब्बा", "gift": "उपहार",
    "card": "कार्ड", "ring": "अंगूठी", "tool": "औज़ार",
    # Adjectives (Adjectives_3of8 .. 8of8)
    "high": "ऊँचा", "low": "नीचा", "soft": "मुलायम", "hard": "सख़्त",
    "deep": "गहरा", "shallow": "उथला", "clean": "साफ़", "dirty": "गंदा",
    "strong": "मज़बूत", "weak": "कमज़ोर", "dead": "मृत", "alive": "जीवित",
    "heavy": "भारी", "light": "हल्का", "famous": "प्रसिद्ध", "long": "लंबा",
    "short": "छोटा", "tall": "ऊँचा", "wide": "चौड़ा", "narrow": "संकरा",
    "biglarge": "बड़ा", "smalllittle": "छोटा", "slow": "धीमा", "fast": "तेज़",
    "hot": "गरम", "cold": "ठंडा", "warm": "गुनगुना", "cool": "ठंडक",
    "new": "नया", "old": "पुराना", "young": "जवान", "bad": "बुरा",
    "wet": "गीला", "dry": "सूखा", "sick": "बीमार", "healthy": "स्वस्थ",
    # People
    "son": "बेटा", "daughter": "बेटी", "mother": "माँ", "father": "पिता",
    "parent": "माता-पिता", "baby": "शिशु", "man": "आदमी", "woman": "औरत",
    "brother": "भाई", "sister": "बहन", "family": "परिवार", "grandfather": "दादा",
    "grandmother": "दादी", "husband": "पति", "wife": "पत्नी", "king": "राजा",
    "queen": "रानी", "president": "राष्ट्रपति", "neighbour": "पड़ोसी", "boy": "लड़का",
    "girl": "लड़की", "child": "बच्चा", "adult": "वयस्क", "friend": "दोस्त",
    "player": "खिलाड़ी", "crowd": "भीड़",
    # Places
    "city": "शहर", "streetorroad": "सड़क", "trainstation": "रेलवे स्टेशन",
    "restaurant": "रेस्टोरेंट", "court": "अदालत", "school": "स्कूल", "office": "दफ़्तर",
    "university": "विश्वविद्यालय", "park": "पार्क", "storeorshop": "दुकान",
    "library": "पुस्तकालय", "hospital": "अस्पताल", "temple": "मंदिर", "market": "बाज़ार",
    "india": "भारत", "ground": "मैदान", "bank": "बैंक", "location": "जगह",
    # Pronouns
    "i": "मैं", "you": "तुम", "he": "वह", "she": "वह", "it": "यह", "we": "हम",
    "youplural": "आप सब", "they": "वे",
    # Seasons
    "summer": "गर्मी", "spring": "वसंत", "winter": "सर्दी", "fall": "पतझड़",
    "season": "मौसम", "monsoon": "मानसून",
    # Society
    "religion": "धर्म", "death": "मृत्यु", "medicine": "दवा", "money": "पैसा",
    "bill": "बिल", "marriage": "शादी", "team": "टीम", "raceethnicity": "जाति",
    "technology": "तकनीक", "energy": "ऊर्जा", "war": "युद्ध", "peace": "शांति",
    "attack": "हमला", "election": "चुनाव", "newspaper": "अख़बार", "gun": "बंदूक",
    "sport": "खेल", "exercise": "व्यायाम", "ball": "गेंद", "price": "क़ीमत",
    "sign": "संकेत", "science": "विज्ञान", "god": "भगवान",
    # Other
    "house": "घर",
}
