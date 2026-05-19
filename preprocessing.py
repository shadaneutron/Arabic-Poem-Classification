
import re
from typing import List

class ArabicTextPreprocessor:
    def __init__(self):
        self.arabic_stopwords = self._get_arabic_stopwords()
    
    def _get_arabic_stopwords(self) -> List[str]:
        return [
            "من", "في", "على", "عن", "إلى", "الذي", "التي", "الذين", "اللواتي",
            "و", "أو", "لكن", "إن", "أن", "ما", "لا", "نعم", "كما", "حيث", "لم",
            "لن", "قد", "لقد", "كان", "كانت", "يكون", "تكون", "هذا", "هذه", "هؤلاء",
            "تلك", "تلكم", "تلكن", "هو", "هي", "نحن", "أنت", "أنتما", "أنتم", "أنتن",
            "هم", "هن", "إيه", "ماذا", "كيف", "متى", "أين", "ليس", "بل", "إلا",
            "حتى", "عند", "بعد", "قبل", "منذ", "كل", "بعض", "أي", "أيها", "أيتها",
            "هكذا", "كذلك", "هنا", "هناك", "حينما", "إذا", "لو", "لولا", "إذ",
            "أو", "أم", "ثم", "أخيرا", "أولئك", "هاؤلاء", "ذو", "ذات", "ذوو",
            "أولاء", "اللذان", "اللتان", "الذين", "اللواتي", "اللائي", "اللتين",
            "ب", "لـ", "كـ", "سـ", "تـ", "يـ", "نـ", "أـ", "ال", "وال", "فال", "بال", "كل",
            "إلى", "عن", "في", "على", "من", "مع", "في", "عن", "إلى", "حتى", "عند",
            "عن", "في", "على", "من", "إلى", "مع", "عن", "في", "على", "من", "إلى",
            "ما", "ماذا", "من", "متي", "أين", "كيف", "لماذا", "أي", "أيه", "أيها",
            "أن", "إن", "إني", "إنا", "أنتم", "أننا", "أنتما", "أنت", "أنا", "نحن",
            "هو", "هي", "هم", "هن", "هوما", "هما", "هاتان", "هذان", "هؤلاء", "أولئك"
        ]
    
    def normalize_alef(self, text: str) -> str:
        return re.sub(r"[إأآاٱ]", "ا", text)
    
    def normalize_yaa(self, text: str) -> str:
        return re.sub(r"[ىيٰ]", "ي", text)
    
    def normalize_taa_marbouta(self, text: str) -> str:
        return re.sub(r"[ة]", "ه", text)
    
    def normalize_hamza(self, text: str) -> str:
        text = re.sub(r"[ؤ]", "و", text)
        text = re.sub(r"[ئ]", "ي", text)
        return text
    
    def normalize_arabic(self, text: str) -> str:
        text = self.normalize_alef(text)
        text = self.normalize_yaa(text)
        text = self.normalize_taa_marbouta(text)
        text = self.normalize_hamza(text)
        return text
    
    def remove_tatweel(self, text: str) -> str:
        return re.sub(r"ـ+", "", text)
    
    def remove_diacritics(self, text: str) -> str:
        diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
        return diacritics.sub("", text)
    
    def remove_non_arabic(self, text: str) -> str:
        return re.sub(r"[^\u0600-\u06FF\s]", "", text)
    
    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
    
    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered_words = [word for word in words if word not in self.arabic_stopwords and len(word) > 1]
        return " ".join(filtered_words)
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        text = text.replace("\n", " ").replace("\r", " ")
        text = self.remove_diacritics(text)
        text = self.remove_tatweel(text)
        text = self.normalize_arabic(text)
        text = self.remove_non_arabic(text)
        text = self.normalize_whitespace(text)
        text = self.remove_stopwords(text)
        
        return text
