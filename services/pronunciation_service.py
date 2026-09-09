"""
Pronunciation Assessment & Speech Therapy Service for meowAI.
Analyzes Vietnamese phonetics, tones, onsets, rhymes, and generates pedagogical feedback.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from services.stt_service import stt_service

logger = logging.getLogger(__name__)

# Vietnamese Tones Mapping
TONE_MAP = {
    'á': 'sắc', 'ắ': 'sắc', 'ấ': 'sắc', 'é': 'sắc', 'ế': 'sắc', 'í': 'sắc',
    'ó': 'sắc', 'ố': 'sắc', 'ớ': 'sắc', 'ú': 'sắc', 'ứ': 'sắc', 'ý': 'sắc',
    
    'à': 'huyền', 'ằ': 'huyền', 'ầ': 'huyền', 'è': 'huyền', 'ề': 'huyền', 'ì': 'huyền',
    'ò': 'huyền', 'ồ': 'huyền', 'ờ': 'huyền', 'ù': 'huyền', 'ừ': 'huyền', 'ỳ': 'huyền',
    
    'ả': 'hỏi', 'ẳ': 'hỏi', 'ẩ': 'hỏi', 'ẻ': 'hỏi', 'ể': 'hỏi', 'ỉ': 'hỏi',
    'ỏ': 'hỏi', 'ổ': 'hỏi', 'ở': 'hỏi', 'ủ': 'hỏi', 'ử': 'hỏi', 'ỷ': 'hỏi',
    
    'ã': 'ngã', 'ẵ': 'ngã', 'ẫ': 'ngã', 'ẽ': 'ngã', 'ễ': 'ngã', 'ĩ': 'ngã',
    'õ': 'ngã', 'ỗ': 'ngã', 'ỡ': 'ngã', 'ũ': 'ngã', 'ữ': 'ngã', 'ỹ': 'ngã',
    
    'ạ': 'nặng', 'ặ': 'nặng', 'ậ': 'nặng', 'ẹ': 'nặng', 'ệ': 'nặng', 'ị': 'nặng',
    'ọ': 'nặng', 'ộ': 'nặng', 'ợ': 'nặng', 'ụ': 'nặng', 'ự': 'nặng', 'ỵ': 'nặng',
}

# Vietnamese Onset Consonants (longest to shortest for greedy match)
ONSET_CONSONANTS = [
    'ngh', 'ng', 'th', 'ch', 'tr', 'nh', 'gh', 'ph', 'kh', 'qu', 'gi',
    'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x'
]


class WordPhonemeScore(BaseModel):
    word: str
    target_word: str
    status: str  # EXCELLENT, GOOD, MISPRONOUNCED, MISSING
    score: float
    tone_correct: bool
    detected_tone: str
    expected_tone: str
    onset_correct: bool
    detected_onset: str
    expected_onset: str
    feedback_note: str


class PronunciationResult(BaseModel):
    reference_text: str
    transcribed_text: str
    overall_score: float
    is_passed: bool
    word_accuracy: float
    tone_accuracy: float
    phoneme_accuracy: float
    duration_seconds: float
    word_details: List[WordPhonemeScore]
    pedagogical_feedback: str


def strip_accents_and_punctuation(text: str) -> str:
    """Normalize text and remove special characters."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_phonetics(word: str) -> Dict[str, str]:
    """
    Deconstruct a single Vietnamese word into Onset, Tone, and Base Rhyme.
    """
    clean_word = word.lower().strip()
    if not clean_word:
        return {'onset': '', 'tone': 'ngang', 'base': ''}

    # 1. Detect Tone
    detected_tone = 'ngang'
    for char, tone_name in TONE_MAP.items():
        if char in clean_word:
            detected_tone = tone_name
            break

    # 2. Detect Onset Consonant (greedy match)
    detected_onset = ''
    for onset in ONSET_CONSONANTS:
        if clean_word.startswith(onset):
            detected_onset = onset
            break

    # 3. Base Rhyme (normalize accents to base)
    normalized = unicodedata.normalize('NFD', clean_word)
    base_chars = [c for c in normalized if unicodedata.category(c) != 'Mn']
    base_word = ''.join(base_chars).replace('đ', 'd')
    rhyme = base_word[len(detected_onset):] if detected_onset else base_word

    return {
        'onset': detected_onset,
        'tone': detected_tone,
        'rhyme': rhyme,
        'raw': clean_word
    }


def compute_word_similarity(target_info: Dict[str, str], detected_info: Dict[str, str]) -> Tuple[float, bool, bool, str]:
    """
    Score similarity between expected word and detected word.
    Returns: (score, tone_correct, onset_correct, note)
    """
    if not detected_info.get('raw'):
        return (0.0, False, False, "Bé chưa đọc từ này")

    # If exact word match
    if target_info['raw'] == detected_info['raw']:
        return (100.0, True, True, "Phát âm chuẩn xác")

    tone_match = (target_info['tone'] == detected_info['tone'])
    onset_match = (target_info['onset'] == detected_info['onset'])

    # Score calculation
    score = 40.0  # Base attempt score
    notes = []

    if tone_match:
        score += 30.0
    else:
        notes.append(f"sai dấu ({detected_info['tone']} thay vì {target_info['tone']})")

    if onset_match:
        score += 30.0
    else:
        det_onset_str = detected_info['onset'] or 'âm zero'
        tgt_onset_str = target_info['onset'] or 'âm zero'
        notes.append(f"ngọng phụ âm đầu '{tgt_onset_str}' thành '{det_onset_str}'")

    note_text = "Cần sửa: " + ", ".join(notes) if notes else "Phát âm khá tốt"
    return (score, tone_match, onset_match, note_text)


class PronunciationAssessmentService:

    async def assess(
        self,
        file_bytes: bytes,
        filename: str,
        reference_text: str,
        language: str = "vi"
    ) -> PronunciationResult:
        """
        Assess child pronunciation from uploaded audio against reference text.
        """
        if not reference_text or reference_text.strip() == "":
            raise ValueError("EMPTY_REFERENCE_TEXT")

        # 1. Transcribe child audio using STT service
        stt_res = await stt_service.transcribe(
            file_bytes=file_bytes,
            filename=filename,
            language=language,
            prompt=reference_text
        )

        transcribed_raw = stt_res.text
        ref_words = strip_accents_and_punctuation(reference_text).split()
        det_words = strip_accents_and_punctuation(transcribed_raw).split()

        word_scores: List[WordPhonemeScore] = []
        tone_correct_count = 0
        onset_correct_count = 0
        total_score_sum = 0.0

        for i, ref_w in enumerate(ref_words):
            target_phon = extract_phonetics(ref_w)
            
            # Find best match or align by index
            if i < len(det_words):
                detected_w = det_words[i]
                detected_phon = extract_phonetics(detected_w)
                score, tone_ok, onset_ok, note = compute_word_similarity(target_phon, detected_phon)
            else:
                detected_w = ""
                detected_phon = {'onset': '', 'tone': '', 'rhyme': '', 'raw': ''}
                score, tone_ok, onset_ok, note = (0.0, False, False, "Bị nuốt/bỏ qua từ này")

            if tone_ok: tone_correct_count += 1
            if onset_ok: onset_correct_count += 1
            total_score_sum += score

            if score >= 90:
                status = "EXCELLENT"
            elif score >= 70:
                status = "GOOD"
            elif score > 0:
                status = "MISPRONOUNCED"
            else:
                status = "MISSING"

            word_scores.append(WordPhonemeScore(
                word=detected_w,
                target_word=ref_w,
                status=status,
                score=score,
                tone_correct=tone_ok,
                detected_tone=detected_phon['tone'],
                expected_tone=target_phon['tone'],
                onset_correct=onset_ok,
                detected_onset=detected_phon['onset'],
                expected_onset=target_phon['onset'],
                feedback_note=note
            ))

        total_words = max(len(ref_words), 1)
        overall_score = round(total_score_sum / total_words, 1)
        tone_accuracy = round((tone_correct_count / total_words) * 100, 1)
        phoneme_accuracy = round((onset_correct_count / total_words) * 100, 1)
        word_accuracy = round((sum(1 for w in word_scores if w.score >= 70) / total_words) * 100, 1)
        is_passed = (overall_score >= 70.0)

        # 3. Generate positive pedagogical feedback
        mispronounced = [w for w in word_scores if w.status in ("MISPRONOUNCED", "MISSING")]
        if overall_score >= 90:
            pedagogical_feedback = "Tuyệt vời! Bé phát âm rất rõ ràng, chuẩn xác từng âm và thanh điệu!"
        elif overall_score >= 70:
            if mispronounced:
                tips = "; ".join(f"từ '{m.target_word}' ({m.feedback_note})" for m in mispronounced[:2])
                pedagogical_feedback = f"Bé làm rất tốt! Chỉ cần chú ý nhẹ ở {tips} là sẽ hoàn hảo!"
            else:
                pedagogical_feedback = "Bé đọc rất hay và tự tin, tiếp tục phát huy nhé!"
        else:
            if mispronounced:
                tips = "; ".join(f"từ '{m.target_word}'" for m in mispronounced[:3])
                pedagogical_feedback = f"Bé đã rất cố gắng! Ba mẹ cùng luyện chậm lại với bé ở {tips} nhé!"
            else:
                pedagogical_feedback = "Bé hãy thử đọc to và rõ ràng hơn một chút cùng cô nhé!"

        return PronunciationResult(
            reference_text=reference_text,
            transcribed_text=transcribed_raw,
            overall_score=overall_score,
            is_passed=is_passed,
            word_accuracy=word_accuracy,
            tone_accuracy=tone_accuracy,
            phoneme_accuracy=phoneme_accuracy,
            duration_seconds=stt_res.duration_seconds,
            word_details=word_scores,
            pedagogical_feedback=pedagogical_feedback
        )


pronunciation_service = PronunciationAssessmentService()
