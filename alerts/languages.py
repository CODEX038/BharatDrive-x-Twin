"""Alert message catalogue — English, Hindi, Marathi. Extend by adding a language dict."""
from __future__ import annotations

MESSAGES = {
    "en": {
        "info_congestion": "Congestion is increasing ahead.",
        "caution_fatigue": "Fatigue indicators are increasing. Stay alert.",
        "high_hazard": "{hazard} {direction}. {advice}",
        "critical": "Driver fatigue and an immediate road hazard are detected. Reduce speed and stop safely.",
        "critical_fatigue": "Severe driver fatigue detected. Please pull over and stop safely.",
        "unreliable": "Camera view is unreliable. Please adjust the camera or lighting.",
        "break": "You have been driving a while and fatigue is rising. Please take a break.",
        "avoid_lane_change": "Avoid changing lanes.",
        "reduce_speed": "Reduce speed gradually.",
    },
    "hi": {
        "info_congestion": "आगे जाम बढ़ रहा है।",
        "caution_fatigue": "थकान के संकेत बढ़ रहे हैं। सतर्क रहें।",
        "high_hazard": "{hazard} {direction}। {advice}",
        "critical": "चालक की थकान और सामने खतरा दोनों पाए गए हैं। गति धीमी करें और सुरक्षित रुकें।",
        "critical_fatigue": "चालक में गंभीर थकान पाई गई है। कृपया गाड़ी किनारे लगाकर सुरक्षित रुकें।",
        "unreliable": "कैमरा दृश्य अविश्वसनीय है। कृपया कैमरा या रोशनी ठीक करें।",
        "break": "आप काफी देर से गाड़ी चला रहे हैं और थकान बढ़ रही है। कृपया विश्राम करें।",
        "avoid_lane_change": "लेन न बदलें।",
        "reduce_speed": "धीरे-धीरे गति कम करें।",
    },
    "mr": {
        "info_congestion": "पुढे वाहतूक कोंडी वाढत आहे.",
        "caution_fatigue": "थकव्याची लक्षणे वाढत आहेत. सावध राहा.",
        "high_hazard": "{hazard} {direction}. {advice}",
        "critical": "चालकाचा थकवा आणि समोर धोका आढळला आहे. वेग कमी करा आणि सुरक्षित थांबा.",
        "critical_fatigue": "चालकामध्ये तीव्र थकवा आढळला आहे. कृपया गाडी बाजूला घेऊन सुरक्षित थांबा.",
        "unreliable": "कॅमेरा दृश्य अविश्वसनीय आहे. कृपया कॅमेरा किंवा प्रकाश समायोजित करा.",
        "break": "आपण बराच वेळ गाडी चालवत आहात आणि थकवा वाढत आहे. कृपया विश्रांती घ्या.",
        "avoid_lane_change": "लेन बदलू नका.",
        "reduce_speed": "हळूहळू वेग कमी करा.",
    },
}


def msg(lang: str, key: str, **kwargs) -> str:
    cat = MESSAGES.get(lang, MESSAGES["en"])
    template = cat.get(key, MESSAGES["en"].get(key, key))
    return template.format(**kwargs) if kwargs else template
