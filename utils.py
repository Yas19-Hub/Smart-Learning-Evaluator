"""
utils.py — Translations, reading accuracy, CSV storage, recommendations.
"""

import csv
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

# ── Data file ──────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_data.csv")

FIELDNAMES = [
    "timestamp", "name", "age", "grade",
    "reading_time", "reading_accuracy", "reaction_time",
    "missed_clicks", "memory_score", "task_completion", "error_rate",
    "prediction", "risk_level", "confidence",
]


def ensure_data_file():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def save_student_data(record: dict):
    ensure_data_file()
    row = {k: record.get(k, "") for k in FIELDNAMES}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def get_previous_record(name: str):
    ensure_data_file()
    matches = []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("name", "").strip().lower() == name.strip().lower():
                    matches.append(row)
    except Exception:
        pass
    return matches[-1] if matches else None


# ── Reading accuracy ───────────────────────────────────────────────────────────

def calculate_reading_accuracy(original: str, spoken: str) -> float:
    """
    Compare spoken text to original using difflib SequenceMatcher.
    - Lowercased
    - Punctuation removed
    - Word-level comparison
    Returns 0.0–100.0
    """
    def clean(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    orig_words   = clean(original)
    spoken_words = clean(spoken)

    if not spoken_words:
        return 0.0
    if not orig_words:
        return 0.0

    ratio = SequenceMatcher(None, orig_words, spoken_words).ratio()
    return round(ratio * 100, 1)


# ── Feature helpers ────────────────────────────────────────────────────────────

def compute_error_rate(missed_clicks: int, total: int = 10) -> float:
    return round((min(missed_clicks, total) / total) * 100, 1)


def compute_task_completion(reading_time: float, reaction_time: float) -> float:
    return round(reading_time + reaction_time / 50.0, 2)


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

READING_PARAGRAPHS = {
    "English": (
        "The sun rises every morning and sets in the evening. "
        "Birds sing beautiful songs in the trees. "
        "Children play happily in the park after school. "
        "Learning new things every day makes us smarter and stronger."
    ),
    "Hindi": (
        "सूरज हर सुबह उगता है और शाम को अस्त होता है। "
        "पेड़ों पर पक्षी मधुर गीत गाते हैं। "
        "बच्चे स्कूल के बाद पार्क में खुशी से खेलते हैं। "
        "हर दिन नई चीजें सीखने से हम और होशियार और मजबूत बनते हैं।"
    ),
}

WORD_SETS = {
    "English": ["Apple", "River", "Cloud", "Music", "Bridge", "Ocean", "Flame", "Pencil"],
    "Hindi":   ["सेब",   "नदी",   "बादल",  "संगीत", "पुल",    "समुद्र", "लौ",   "पेंसिल"],
}

# Speech recognition language codes
SPEECH_LANG = {
    "English": "en-US",
    "Hindi":   "hi-IN",
}

IMAGE_MATCH_DATA = {
    "English": {
        "sun.png":  ["Sun", "Son", "Bun", "Run"],
        "tree.png": ["Tree", "Free", "Three", "Tee"],
        "cat.png":  ["Cat", "Bat", "Cot", "Tac"],
    },
    "Hindi": {
        "sun.png":  ["सूरज", "मूरत", "पूरब", "सरत"],
        "tree.png": ["पेड़", "भीड़", "भेड़", "पड़"],
        "cat.png":  ["बिल्ली", "तितली", "बिल्ला", "किल्ली"],
    }
}

MEMORY_IMAGES = {
    "English": {"sun.png": "sun", "tree.png": "tree", "cat.png": "cat", "apple.png": "apple", "car.png": "car"},
    "Hindi":   {"sun.png": "सूरज", "tree.png": "पेड़", "cat.png": "बिल्ली", "apple.png": "सेब", "car.png": "कार"}
}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS  (every UI string in one place)
# ══════════════════════════════════════════════════════════════════════════════

T = {
    # ── English ───────────────────────────────────────────────────────────────
    "English": {
        # Global
        "lang_label":        "🌐 Language",
        "app_name":          "Revon",
        "tagline":           "AI-Powered Learning Disability Detection",
        "quote":             '"Every child learns differently. Revon helps you understand how."',
        "get_started":       "Get Started →",
        "next":              "Continue →",
        "start_over":        "🔄 Start Over",
        "step_label":        "Step",
        "of_label":          "of",
        "step_names":        ["Student Info", "Attention", "Reading", "Word Match", "Memory", "Results"],
        # Form
        "student_info":      "Student Information",
        "full_name":         "Full Name",
        "age":               "Age",
        "grade":             "Grade / Class",
        # Attention
        "attention_title":   "🎯 Attention Test",
        "attention_inst":    "Click the glowing 🎯 target as fast as you can. 10 rounds total.",
        "start_test":        "▶  Start Attention Test",
        "attn_done_msg":     "✅ Attention test complete!",
        "round_label":       "Round",
        "hits_label":        "Hits",
        "missed_label":      "Missed",
        "avg_rt_label":      "Avg Reaction Time",
        # Reading
        "reading_title":     "📖 Reading Test",
        "reading_inst":      "Read the paragraph below aloud. Press Start Recording, read clearly, then press Stop.",
        "reading_para_lbl":  "📄 Read this paragraph aloud:",
        "start_recording":   "🎙️  Start Recording",
        "stop_recording":    "⏹️  Stop Recording",
        "recording_status":  "🔴 Recording… read the paragraph aloud now!",
        "mic_unsupported":   "⚠️ Your browser does not support speech recognition. Please use Chrome or Edge.",
        "mic_error":         "⚠️ Microphone error",
        "transcript_label":  "📝 Transcribed text:",
        "transcript_empty":  "Your spoken words will appear here…",
        "confirm_reading":   "✅ Confirm & Calculate Accuracy",
        "reading_accuracy":  "Reading Accuracy",
        "reading_time":      "Reading Time",
        "read_again":        "🔄 Read Again",
        "no_speech_warn":    "⚠️ No speech detected. Please record again.",
        # Memory
        "memory_title":      "🧠 Memory Test",
        "memory_inst":       "Memorise the 5 words below, then click Hide and recall them.",
        "hide_recall":       "Hide Words & Start Recall",
        "recall_inst":       "Type the words you remember (comma-separated):",
        "recall_ph":         "e.g.  Apple, River, Cloud",
        "submit_recall":     "Submit Recall",
        "memory_score":      "Memory Score",
        # Image Match
        "image_match_title": "🖼️ Word Match Test",
        "image_match_inst":  "Look at the picture closely. Which word matches the picture?",
        "image_match_done":  "✅ Good job! Word match complete.",
        "image_score":       "Word Match Score",
        # Results
        "view_results":      "View My Results →",
        "results_title":     "Assessment Results",
        "prediction":        "Prediction",
        "risk_level":        "Risk Level",
        "confidence":        "Confidence",
        "attention_score":   "Attention Score",
        "reaction_time":     "Reaction Time",
        "missed_clicks":     "Missed Clicks",
        "insights":          "💡 Insights",
        "recommendations":   "✅ Recommendations",
        "next_steps":        "🚀 Next Steps",
        "prob_breakdown":    "Class Probability Breakdown",
        "debug_title":       "🔧 Debug — Feature Values",
        "prev_session":      "Previous Session",
        "curr_session":      "Current Session",
        "impr_read_up":      "🎉 Reading accuracy improved since last session!",
        "impr_read_dn":      "📖 Reading accuracy dropped — keep practising.",
        "impr_attn_up":      "✨ Attention has improved!",
        "impr_attn_dn":      "🧠 Attention needs more work.",
        "disclaimer":        (
            "⚠️ Disclaimer: Revon is a screening tool only — not a clinical diagnosis. "
            "Results should be reviewed by a qualified educational psychologist."
        ),
    },

    # ── Hindi ─────────────────────────────────────────────────────────────────
    "Hindi": {
        # Global
        "lang_label":        "🌐 भाषा",
        "app_name":          "Revon",
        "tagline":           "AI-आधारित लर्निंग डिसेबिलिटी डिटेक्शन",
        "quote":             '"हर बच्चा अलग तरह से सीखता है। Revon आपको समझने में मदद करता है।"',
        "get_started":       "शुरू करें →",
        "next":              "आगे बढ़ें →",
        "start_over":        "🔄 फिर से शुरू करें",
        "step_label":        "चरण",
        "of_label":          "का",
        "step_names":        ["छात्र जानकारी", "ध्यान", "पठन", "शब्द मिलान", "स्मृति", "परिणाम"],
        # Form
        "student_info":      "छात्र की जानकारी",
        "full_name":         "पूरा नाम",
        "age":               "आयु",
        "grade":             "कक्षा",
        # Attention
        "attention_title":   "🎯 ध्यान परीक्षण",
        "attention_inst":    "चमकते 🎯 लक्ष्य पर जितनी जल्दी हो सके क्लिक करें। कुल 10 राउंड।",
        "start_test":        "▶  ध्यान परीक्षण शुरू करें",
        "attn_done_msg":     "✅ ध्यान परीक्षण पूरा हुआ!",
        "round_label":       "राउंड",
        "hits_label":        "हिट",
        "missed_label":      "चूका",
        "avg_rt_label":      "औसत प्रतिक्रिया समय",
        # Reading
        "reading_title":     "📖 पठन परीक्षण",
        "reading_inst":      "नीचे दिया गया अनुच्छेद ज़ोर से पढ़ें। रिकॉर्डिंग शुरू करें, स्पष्ट रूप से पढ़ें, फिर रोकें।",
        "reading_para_lbl":  "📄 यह अनुच्छेद ज़ोर से पढ़ें:",
        "start_recording":   "🎙️  रिकॉर्डिंग शुरू करें",
        "stop_recording":    "⏹️  रिकॉर्डिंग बंद करें",
        "recording_status":  "🔴 रिकॉर्डिंग हो रही है… अभी अनुच्छेद ज़ोर से पढ़ें!",
        "mic_unsupported":   "⚠️ आपका ब्राउज़र स्पीच रिकग्निशन को सपोर्ट नहीं करता। कृपया Chrome या Edge उपयोग करें।",
        "mic_error":         "⚠️ माइक्रोफ़ोन त्रुटि",
        "transcript_label":  "📝 लिखित पाठ:",
        "transcript_empty":  "आपके बोले गए शब्द यहाँ दिखेंगे…",
        "confirm_reading":   "✅ पुष्टि करें और सटीकता मापें",
        "reading_accuracy":  "पठन सटीकता",
        "reading_time":      "पठन समय",
        "read_again":        "🔄 फिर से पढ़ें",
        "no_speech_warn":    "⚠️ कोई भाषण नहीं पहचाना गया। कृपया फिर से रिकॉर्ड करें।",
        # Memory
        "memory_title":      "🧠 स्मृति परीक्षण",
        "memory_inst":       "नीचे दिए 5 शब्दों को याद करें, फिर छुपाएं और याद करें।",
        "hide_recall":       "शब्द छुपाएं और याद करें",
        "recall_inst":       "याद किए गए शब्द लिखें (अल्पविराम से अलग करें):",
        "recall_ph":         "जैसे: सेब, नदी, बादल",
        "submit_recall":     "उत्तर जमा करें",
        "memory_score":      "स्मृति स्कोर",
        # Image Match
        "image_match_title": "🖼️ शब्द मिलान परीक्षण",
        "image_match_inst":  "चित्र को ध्यान से देखें। कौन सा शब्द चित्र से मेल खाता है?",
        "image_match_done":  "✅ बहुत बढ़िया! शब्द मिलान पूरा हुआ।",
        "image_score":       "शब्द मिलान स्कोर",
        # Results
        "view_results":      "मेरे परिणाम देखें →",
        "results_title":     "मूल्यांकन परिणाम",
        "prediction":        "पूर्वानुमान",
        "risk_level":        "जोखिम स्तर",
        "confidence":        "विश्वास स्तर",
        "attention_score":   "ध्यान स्कोर",
        "reaction_time":     "प्रतिक्रिया समय",
        "missed_clicks":     "छूटे क्लिक",
        "insights":          "💡 अंतर्दृष्टि",
        "recommendations":   "✅ सिफारिशें",
        "next_steps":        "🚀 अगले कदम",
        "prob_breakdown":    "वर्ग संभावना विश्लेषण",
        "debug_title":       "🔧 डिबग — फीचर वैल्यू",
        "prev_session":      "पिछला सत्र",
        "curr_session":      "वर्तमान सत्र",
        "impr_read_up":      "🎉 पिछले सत्र से पठन सटीकता में सुधार हुआ!",
        "impr_read_dn":      "📖 पठन सटीकता कम हुई — अभ्यास जारी रखें।",
        "impr_attn_up":      "✨ ध्यान में सुधार हुआ!",
        "impr_attn_dn":      "🧠 ध्यान को और सुधार की जरूरत है।",
        "disclaimer":        (
            "⚠️ अस्वीकरण: Revon केवल एक स्क्रीनिंग टूल है — यह कोई चिकित्सकीय निदान नहीं है। "
            "परिणामों की समीक्षा एक योग्य शैक्षिक मनोवैज्ञानिक द्वारा की जानी चाहिए।"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS  (insights / recs / next steps per prediction × language)
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "English": {
        "Normal": {
            "insights": [
                "Cognitive performance across all tested areas is within the healthy range.",
                "Reading fluency and comprehension appear well-developed for this age.",
                "Attention and memory scores indicate a strong learning capacity.",
            ],
            "recommendations": [
                "Keep encouraging daily reading — even 15 minutes makes a lasting difference.",
                "Engage in memory games, puzzles, and strategy games regularly.",
                "Celebrate progress and maintain a positive, low-stress learning environment.",
            ],
            "next_steps": [
                "Schedule a routine check-up in 6 months to track any changes.",
                "Explore enrichment activities and advanced reading programmes.",
                "Monitor school performance across subjects for consistency.",
            ],
        },
        "Dyslexia Risk": {
            "insights": [
                "Assessment indicates possible challenges with reading fluency and accuracy.",
                "Longer reading time and lower accuracy are key indicators observed.",
                "Dyslexia does not affect intelligence — it is a processing difference, not a deficit.",
            ],
            "recommendations": [
                "Consult a certified educational psychologist or dyslexia specialist.",
                "Use text-to-speech tools, audiobooks, and dyslexia-friendly fonts.",
                "Practice phonics-based reading exercises for 10–15 minutes daily.",
                "Try coloured overlays or pastel-tinted paper to ease visual stress.",
            ],
            "next_steps": [
                "Request a formal dyslexia assessment from a qualified specialist.",
                "Inform the school for possible accommodations (extra time in exams).",
                "Explore structured literacy programmes such as Orton-Gillingham.",
            ],
        },
        "ADHD Risk": {
            "insights": [
                "Assessment suggests possible challenges with sustained attention and impulse control.",
                "Higher missed clicks and variable reaction times were observed.",
                "ADHD is highly manageable with the right structure and professional support.",
            ],
            "recommendations": [
                "Consult a paediatric neurologist or child psychologist for evaluation.",
                "Create consistent daily routines and minimise distractions at study time.",
                "Use short, focused study sessions (15–20 min) with movement breaks.",
                "Reward consistency and celebrate small, incremental achievements.",
            ],
            "next_steps": [
                "Seek a professional ADHD evaluation — a diagnosis opens access to support.",
                "Discuss accommodations with the school (preferential seating, extra time).",
                "Explore behavioural therapy or mindfulness techniques.",
            ],
        },
        "Learning Difficulty": {
            "insights": [
                "Multiple cognitive areas show signs of learning challenges.",
                "Memory, reading accuracy, and attention all indicate the need for targeted support.",
                "Early, targeted intervention can make a significant positive difference.",
            ],
            "recommendations": [
                "Work with a multidisciplinary team: teacher, psychologist, and therapist.",
                "Use multi-sensory teaching approaches (visual, auditory, and tactile).",
                "Break all tasks into small, clearly-defined steps with frequent check-ins.",
                "Build confidence through strengths-based activities alongside support work.",
            ],
            "next_steps": [
                "Commission a comprehensive educational psychology assessment.",
                "Develop an Individualised Education Plan (IEP) with the school.",
                "Connect with local support organisations and parent networks.",
            ],
        },
    },
    "Hindi": {
        "Normal": {
            "insights": [
                "सभी परीक्षित क्षेत्रों में संज्ञानात्मक प्रदर्शन स्वस्थ सीमा में है।",
                "पठन प्रवाह और समझ इस आयु के लिए अच्छी तरह विकसित प्रतीत होती है।",
                "ध्यान और स्मृति स्कोर मजबूत सीखने की क्षमता दर्शाते हैं।",
            ],
            "recommendations": [
                "प्रतिदिन पढ़ने को प्रोत्साहित करते रहें — 15 मिनट भी बड़ा फर्क डालते हैं।",
                "स्मृति खेल, पहेलियाँ और रणनीति खेल नियमित रूप से खेलें।",
                "सकारात्मक और कम तनाव वाला सीखने का वातावरण बनाए रखें।",
            ],
            "next_steps": [
                "6 महीने में नियमित जांच करें।",
                "उन्नत पठन और समृद्धि कार्यक्रमों में भाग लें।",
                "स्कूल के विषयों में प्रदर्शन पर नज़र रखें।",
            ],
        },
        "Dyslexia Risk": {
            "insights": [
                "पठन प्रवाह और सटीकता में संभावित चुनौतियाँ हैं।",
                "अधिक पठन समय और कम सटीकता मुख्य संकेतक हैं।",
                "डिस्लेक्सिया बुद्धि को प्रभावित नहीं करता — यह एक प्रसंस्करण अंतर है।",
            ],
            "recommendations": [
                "डिस्लेक्सिया विशेषज्ञ या शैक्षिक मनोवैज्ञानिक से परामर्श करें।",
                "टेक्स्ट-टू-स्पीच टूल, ऑडियोबुक और डिस्लेक्सिया-अनुकूल फ़ॉन्ट का उपयोग करें।",
                "प्रतिदिन 10-15 मिनट फोनिक्स-आधारित पठन अभ्यास करें।",
            ],
            "next_steps": [
                "विशेषज्ञ से औपचारिक डिस्लेक्सिया मूल्यांकन करवाएं।",
                "स्कूल को परीक्षा में अतिरिक्त समय के लिए सूचित करें।",
                "संरचित साक्षरता कार्यक्रमों का अन्वेषण करें।",
            ],
        },
        "ADHD Risk": {
            "insights": [
                "ध्यान और आवेग नियंत्रण में संभावित चुनौतियाँ हैं।",
                "अधिक मिस्ड क्लिक और परिवर्तनशील प्रतिक्रिया समय देखा गया।",
                "सही रणनीतियों और पेशेवर सहायता से ADHD को प्रबंधित किया जा सकता है।",
            ],
            "recommendations": [
                "बाल मनोवैज्ञानिक या न्यूरोलॉजिस्ट से परामर्श करें।",
                "संरचित दिनचर्या बनाएं और अध्ययन के समय ध्यान भटकाने वाली चीजें कम करें।",
                "15-20 मिनट के छोटे, केंद्रित अध्ययन सत्रों का उपयोग करें।",
            ],
            "next_steps": [
                "पेशेवर ADHD मूल्यांकन लें।",
                "स्कूल के साथ संभावित सुविधाओं पर चर्चा करें।",
                "व्यवहार चिकित्सा या माइंडफुलनेस तकनीकों का पता लगाएं।",
            ],
        },
        "Learning Difficulty": {
            "insights": [
                "कई संज्ञानात्मक क्षेत्रों में सीखने की चुनौतियाँ हैं।",
                "स्मृति, पठन सटीकता और ध्यान सभी को लक्षित सहायता की आवश्यकता है।",
                "शीघ्र, लक्षित हस्तक्षेप महत्वपूर्ण सकारात्मक बदलाव ला सकता है।",
            ],
            "recommendations": [
                "बहु-विषयक टीम के साथ काम करें: शिक्षक, मनोवैज्ञानिक और थेरेपिस्ट।",
                "बहु-संवेदी शिक्षण दृष्टिकोण का उपयोग करें (दृश्य, श्रवण और स्पर्श)।",
                "कार्यों को छोटे, स्पष्ट चरणों में विभाजित करें।",
            ],
            "next_steps": [
                "शैक्षिक मनोवैज्ञानिक के साथ व्यापक मूल्यांकन करें।",
                "स्कूल के साथ व्यक्तिगत शिक्षा योजना (IEP) विकसित करें।",
                "सहायता संगठनों और माता-पिता नेटवर्क से जुड़ें।",
            ],
        },
    },
}
