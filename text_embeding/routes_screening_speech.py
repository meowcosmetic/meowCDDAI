from typing import Optional, List, Dict, Any
import logging
import os
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import sys
import subprocess
import tempfile

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

# Lazy import librosa
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("[Speech] librosa không được cài đặt. Vui lòng cài: pip install librosa soundfile")
    librosa = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("[Speech] matplotlib không được cài đặt. Visualization sẽ bị tắt.")
    plt = None

router = APIRouter(prefix="/screening/speech", tags=["Screening - Speech & Audio"])


def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio từ video file sử dụng moviepy hoặc ffmpeg
    """
    # Ưu tiên dùng moviepy (dễ hơn trên Windows)
    try:
        from moviepy.editor import VideoFileClip
        logger.info(f"[Speech] Sử dụng moviepy để extract audio từ video")
        
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            raise HTTPException(
                status_code=400,
                detail="Video không có audio track"
            )
        
        # Write audio file với format wav
        video.audio.write_audiofile(
            output_audio_path,
            verbose=False,
            logger=None,
            codec='pcm_s16le',  # PCM 16-bit
            fps=22050  # Sample rate
        )
        video.close()
        
        if os.path.exists(output_audio_path):
            logger.info(f"[Speech] ✅ Đã extract audio thành công: {output_audio_path}")
            return True
        else:
            raise Exception("Audio file không được tạo")
            
    except ImportError:
        # Fallback: thử dùng ffmpeg
        logger.info("[Speech] moviepy không có, thử dùng ffmpeg")
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                output_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"ffmpeg failed: {result.stderr}")
            
            if os.path.exists(output_audio_path):
                logger.info(f"[Speech] ✅ Đã extract audio thành công với ffmpeg: {output_audio_path}")
                return True
            else:
                raise Exception("Audio file không được tạo")
                
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Cần cài đặt moviepy hoặc ffmpeg để extract audio từ video. "
                       "Cài: pip install moviepy"
            )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Speech] Lỗi khi extract audio: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể extract audio từ video: {error_msg}"
        )


def draw_speech_annotations_on_video(frame, current_time, voice_activity_frame, 
                                    vocalizations_at_time, babbling_detected, 
                                    speech_percentage, frame_count=0, fps=30,
                                    child_segments=None, adult_segments=None,
                                    child_speech_percentage=0.0, adult_speech_percentage=0.0):
    """
    Vẽ annotations lên video frame bao gồm thông tin phân loại giọng nói
    """
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Status bar ở trên (tăng chiều cao để chứa thêm thông tin)
    status_bar_height = 140
    cv2.rectangle(annotated_frame, (0, 0), (w, status_bar_height), (0, 0, 0), -1)
    
    # Voice activity indicator (dòng 1)
    if voice_activity_frame:
        status_color = (0, 255, 0)  # Xanh lá
        status_text = "SPEECH DETECTED"
    else:
        status_color = (0, 0, 255)  # Đỏ
        status_text = "SILENCE"
    
    cv2.putText(annotated_frame, status_text, (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Speech percentage (dòng 2)
    cv2.putText(annotated_frame, f"Speech: {speech_percentage:.1f}%", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Child/Adult speech classification (dòng 3-4)
    current_child_segment = None
    current_adult_segment = None
    
    if child_segments is not None and adult_segments is not None:
        # Tìm segment hiện tại
        for seg in child_segments:
            if seg['start_time'] <= current_time <= seg['end_time']:
                current_child_segment = seg
                break
        
        for seg in adult_segments:
            if seg['start_time'] <= current_time <= seg['end_time']:
                current_adult_segment = seg
                break
        
        # Hiển thị thông tin speaker hiện tại (dòng 3)
        if current_child_segment:
            cv2.putText(annotated_frame, "CHILD SPEECH", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Vàng
            if current_child_segment['pitch_mean']:
                cv2.putText(annotated_frame, f"Pitch: {current_child_segment['pitch_mean']:.0f} Hz", (20, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif current_adult_segment:
            cv2.putText(annotated_frame, "ADULT SPEECH", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Cam
            if current_adult_segment['pitch_mean']:
                cv2.putText(annotated_frame, f"Pitch: {current_adult_segment['pitch_mean']:.0f} Hz", (20, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Hiển thị tổng quan ở góc phải (dòng 2-3)
        cv2.putText(annotated_frame, f"Child: {child_speech_percentage:.1f}%", (w - 200, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated_frame, f"Adult: {adult_speech_percentage:.1f}%", (w - 200, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Babbling indicator (dòng 4, chỉ hiển thị nếu không có child/adult info)
    if not current_child_segment and not current_adult_segment:
        if babbling_detected:
            cv2.putText(annotated_frame, "BABBLING: YES", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "BABBLING: NO", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    else:
        # Hiển thị babbling ở góc phải dòng 4
        if babbling_detected:
            cv2.putText(annotated_frame, "BABBLING: YES", (w - 200, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Vocalizations tại thời điểm này (góc phải dòng 1)
    if vocalizations_at_time:
        cv2.putText(annotated_frame, f"Vocalizations: {len(vocalizations_at_time)}", (w - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Time và frame info ở dưới
    time_text = f"Time: {current_time:.2f}s | Frame: {frame_count}"
    cv2.rectangle(annotated_frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(annotated_frame, time_text, (10, h - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Vẽ waveform visualization nhỏ ở góc dưới bên phải
    if voice_activity_frame:
        # Vẽ indicator bar
        bar_width = 50
        bar_height = 20
        bar_x = w - bar_width - 10
        bar_y = h - bar_height - 40
        
        # Màu khác nhau cho child/adult
        if child_segments is not None and adult_segments is not None:
            if current_child_segment:
                bar_color = (0, 255, 255)  # Vàng cho trẻ em
            elif current_adult_segment:
                bar_color = (255, 165, 0)  # Cam cho người lớn
            else:
                bar_color = status_color
        else:
            bar_color = status_color
        
        cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), bar_color, -1)
        cv2.putText(annotated_frame, "VOICE", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
    
    return annotated_frame


def detect_voice_activity(y, sr, frame_length=2048, hop_length=512, energy_threshold=0.01):
    """
    Voice Activity Detection (VAD) dựa trên energy
    """
    # Tính energy cho mỗi frame
    frame_length_samples = frame_length
    hop_length_samples = hop_length
    
    # Tính RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
    
    # Normalize energy
    energy_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
    
    # Voice activity: energy > threshold
    voice_activity = energy_normalized > energy_threshold
    
    # Tính thời gian voice activity
    frames_per_second = sr / hop_length_samples
    voice_frames = np.sum(voice_activity)
    voice_duration = voice_frames / frames_per_second
    
    return voice_activity, voice_duration, rms


def detect_vocalizations(y, sr, min_duration=0.1, max_duration=2.0, energy_threshold=0.01):
    """
    Detect các vocalization events (phát âm ngắn)
    """
    voice_activity, _, rms = detect_voice_activity(y, sr, energy_threshold=energy_threshold)
    
    # Tìm các segments liên tục
    vocalizations = []
    in_vocalization = False
    start_frame = 0
    
    frames_per_second = sr / 512
    
    for i, is_voice in enumerate(voice_activity):
        if is_voice and not in_vocalization:
            # Bắt đầu vocalization
            in_vocalization = True
            start_frame = i
        elif not is_voice and in_vocalization:
            # Kết thúc vocalization
            in_vocalization = False
            duration = (i - start_frame) / frames_per_second
            
            if min_duration <= duration <= max_duration:
                vocalizations.append({
                    'start_time': start_frame / frames_per_second,
                    'duration': duration,
                    'end_time': i / frames_per_second
                })
    
    # Xử lý vocalization cuối cùng nếu còn đang active
    if in_vocalization:
        duration = (len(voice_activity) - start_frame) / frames_per_second
        if min_duration <= duration <= max_duration:
            vocalizations.append({
                'start_time': start_frame / frames_per_second,
                'duration': duration,
                'end_time': len(voice_activity) / frames_per_second
            })
    
    return vocalizations


def detect_babbling(y, sr):
    """
    Detect babbling patterns dựa trên frequency analysis
    Babbling thường có:
    - Frequency range: 200-3000 Hz (voice range của trẻ)
    - Repetitive patterns
    - Moderate energy
    """
    # Tính spectrogram
    stft = librosa.stft(y, hop_length=512)
    magnitude = np.abs(stft)
    
    # Tính frequency bins
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Tìm frequency range của voice (200-3000 Hz)
    voice_freq_mask = (frequencies >= 200) & (frequencies <= 3000)
    voice_energy = np.mean(magnitude[voice_freq_mask, :], axis=0)
    
    # Tính energy threshold
    energy_threshold = np.percentile(voice_energy, 30)
    
    # Detect babbling: có energy trong voice range và có variation
    voice_frames = voice_energy > energy_threshold
    voice_percentage = np.sum(voice_frames) / len(voice_frames) * 100
    
    # Tính variation (coefficient of variation)
    if np.mean(voice_energy) > 0:
        variation = np.std(voice_energy) / np.mean(voice_energy)
    else:
        variation = 0
    
    # Babbling: có voice energy và có variation (repetitive patterns)
    is_babbling = voice_percentage > 10 and variation > 0.3
    
    return is_babbling, voice_percentage, variation


def extract_pitch_features(y, sr, hop_length=512):
    """
    Extract pitch (F0) features để phân biệt trẻ em và người lớn
    Trẻ em thường có pitch cao hơn (300-500 Hz) so với người lớn (100-300 Hz)
    Giọng nữ trưởng thành: 180-300 Hz, giọng nam: 100-200 Hz
    """
    try:
        # Sử dụng librosa pyin để tính pitch chính xác hơn
        # pyin cho kết quả tốt hơn piptrack
        f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                      fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                                                      fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                                                      sr=sr,
                                                      hop_length=hop_length,
                                                      threshold=0.1)
        
        # Lọc chỉ lấy các giá trị pitch hợp lệ (không phải NaN)
        pitch_values = f0[~np.isnan(f0)]
        
        if len(pitch_values) == 0:
            # Fallback: dùng piptrack nếu pyin không hoạt động
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                return None, None, None
            
            pitch_values = np.array(pitch_values)
        
        # Tính các thống kê
        mean_pitch = np.mean(pitch_values)
        median_pitch = np.median(pitch_values)
        std_pitch = np.std(pitch_values)
        
        return mean_pitch, median_pitch, std_pitch
    except Exception as e:
        logger.warning(f"[Speech] Không thể extract pitch: {str(e)}")
        return None, None, None


def extract_formant_features(y, sr, hop_length=512):
    """
    Extract formant frequencies (F1, F2, F3) để phân biệt trẻ em và người lớn
    Trẻ em thường có formants cao hơn do vocal tract nhỏ hơn
    """
    try:
        # Tính spectrogram
        stft = librosa.stft(y, hop_length=hop_length, n_fft=2048)
        magnitude = np.abs(stft)
        
        # Tính frequency bins
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Tìm formants (peaks trong spectrum)
        # F1: 300-1000 Hz, F2: 1000-3000 Hz, F3: 2500-4000 Hz
        formant_ranges = [
            (300, 1000),   # F1
            (1000, 3000),  # F2
            (2500, 4000)   # F3
        ]
        
        formants = []
        for f_min, f_max in formant_ranges:
            freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
            if np.any(freq_mask):
                # Tìm peak trong range này
                energy_in_range = np.mean(magnitude[freq_mask, :], axis=0)
                peak_idx = np.argmax(energy_in_range)
                peak_freq = frequencies[freq_mask][np.argmax(magnitude[freq_mask, peak_idx])]
                formants.append(peak_freq)
            else:
                formants.append(0)
        
        return formants if len(formants) == 3 else None
    except Exception as e:
        logger.warning(f"[Speech] Không thể extract formants: {str(e)}")
        return None


def classify_voice_age(pitch_mean, pitch_median, formants=None):
    """
    Phân loại giọng nói là trẻ em hay người lớn dựa trên pitch và formants
    
    Args:
        pitch_mean: Mean pitch (Hz)
        pitch_median: Median pitch (Hz)
        formants: List [F1, F2, F3] hoặc None
    
    Returns:
        'child', 'adult', hoặc 'unknown'
    
    Note:
        - Trẻ em (2-10 tuổi): pitch thường 300-500 Hz
        - Giọng nữ trưởng thành: 180-300 Hz
        - Giọng nam trưởng thành: 100-200 Hz
        - Trẻ sơ sinh: pitch có thể > 400 Hz
    """
    if pitch_mean is None or pitch_median is None:
        return 'unknown'
    
    # Điều chỉnh thresholds để chính xác hơn:
    # - Trẻ em: pitch > 320 Hz (cao hơn để tránh nhầm với giọng nữ)
    # - Người lớn: pitch < 280 Hz
    # - Vùng giữa (280-320 Hz): cần phân tích thêm với formants
    
    child_threshold_high = 320  # Hz - chắc chắn là trẻ em
    child_threshold_low = 280    # Hz - có thể là trẻ em
    adult_threshold = 280        # Hz - có thể là người lớn
    
    # Phân loại dựa trên pitch
    if pitch_mean >= child_threshold_high:
        # Pitch rất cao (> 320 Hz) - chắc chắn là trẻ em
        return 'child'
    elif pitch_mean <= adult_threshold:
        # Pitch thấp (< 280 Hz) - có thể là người lớn
        # Kiểm tra thêm với formants để chắc chắn
        if formants and len(formants) == 3:
            f1, f2, f3 = formants
            # Người lớn thường có formants thấp hơn
            if f1 < 550 and f2 < 1900:
                return 'adult'
            # Nếu formants cao nhưng pitch thấp, có thể là giọng nữ trẻ
            elif pitch_mean < 200:
                return 'adult'
        
        return 'adult'
    else:
        # Vùng giữa (280-320 Hz) - cần phân tích kỹ hơn
        if formants and len(formants) == 3:
            f1, f2, f3 = formants
            # Trẻ em thường có formants cao hơn
            if f1 > 600 and f2 > 2000:
                return 'child'
            elif f1 < 550 and f2 < 1900:
                return 'adult'
        
        # Nếu không có formants, dựa vào pitch median
        if pitch_median and pitch_median > 300:
            return 'child'
        else:
            return 'adult'


def detect_speakers_and_classify(y, sr, voice_activity, hop_length=512):
    """
    Phân tách các speaker và phân loại từng segment là trẻ em hay người lớn
    
    Returns:
        List of segments với thông tin speaker và age classification
    """
    frames_per_second = sr / hop_length
    
    # Tìm các speech segments liên tục
    speech_segments = []
    in_speech = False
    start_frame = 0
    
    for i, is_voice in enumerate(voice_activity):
        if is_voice and not in_speech:
            in_speech = True
            start_frame = i
        elif not is_voice and in_speech:
            in_speech = False
            end_frame = i
            duration = (end_frame - start_frame) / frames_per_second
            
            # Chỉ xử lý segments có duration >= 0.2 giây
            if duration >= 0.2:
                start_time = start_frame / frames_per_second
                end_time = end_frame / frames_per_second
                
                # Extract audio segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = y[start_sample:end_sample]
                
                # Extract features
                pitch_mean, pitch_median, pitch_std = extract_pitch_features(segment_audio, sr, hop_length)
                formants = extract_formant_features(segment_audio, sr, hop_length)
                
                # Classify
                voice_type = classify_voice_age(pitch_mean, pitch_median, formants)
                
                speech_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'voice_type': voice_type,  # 'child', 'adult', 'unknown'
                    'pitch_mean': float(pitch_mean) if pitch_mean else None,
                    'pitch_median': float(pitch_median) if pitch_median else None,
                    'formants': formants if formants else None
                })
    
    # Xử lý segment cuối cùng nếu còn đang active
    if in_speech:
        end_frame = len(voice_activity)
        duration = (end_frame - start_frame) / frames_per_second
        if duration >= 0.2:
            start_time = start_frame / frames_per_second
            end_time = end_frame / frames_per_second
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            pitch_mean, pitch_median, pitch_std = extract_pitch_features(segment_audio, sr, hop_length)
            formants = extract_formant_features(segment_audio, sr, hop_length)
            voice_type = classify_voice_age(pitch_mean, pitch_median, formants)
            
            speech_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'voice_type': voice_type,
                'pitch_mean': float(pitch_mean) if pitch_mean else None,
                'pitch_median': float(pitch_median) if pitch_median else None,
                'formants': formants if formants else None
            })
    
    return speech_segments


def create_audio_visualization(y, sr, voice_activity, vocalizations, output_path=None):
    """
    Tạo visualization cho audio analysis
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 1. Waveform với voice activity overlay
        time_axis = np.linspace(0, len(y) / sr, len(y))
        axes[0].plot(time_axis, y, alpha=0.7, label='Audio')
        
        # Vẽ voice activity regions
        frames_per_second = sr / 512
        voice_time = np.arange(len(voice_activity)) / frames_per_second
        axes[0].fill_between(voice_time, -1, 1, where=voice_activity, 
                            alpha=0.3, color='green', label='Voice Activity')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Waveform với Voice Activity Detection')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        stft = librosa.stft(y, hop_length=512)
        magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(magnitude_db, sr=sr, hop_length=512, 
                                x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Spectrogram')
        axes[1].set_ylabel('Frequency (Hz)')
        
        # 3. Vocalizations timeline
        axes[2].set_xlim(0, len(y) / sr)
        axes[2].set_ylim(-0.5, 1.5)
        for v in vocalizations:
            axes[2].barh(0, v['duration'], left=v['start_time'], 
                         height=0.5, color='blue', alpha=0.7)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Vocalizations')
        axes[2].set_title(f'Vocalizations ({len(vocalizations)} events)')
        axes[2].set_yticks([0])
        axes[2].set_yticklabels(['Vocalizations'])
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            # Return figure as bytes
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
    except Exception as e:
        logger.warning(f"[Speech] Không thể tạo visualization: {str(e)}")
        return None


class SpeechAnalysisResponse(BaseModel):
    """Response model cho phân tích speech/audio"""
    audio_duration: float = Field(..., description="Thời lượng audio (giây)")
    speech_duration: float = Field(..., description="Thời lượng có tiếng nói (giây)")
    speech_percentage: float = Field(..., description="Phần trăm thời gian có tiếng nói (%)")
    vocalization_frequency: float = Field(..., description="Tần suất phát âm (số lần/giây)")
    silence_percentage: float = Field(..., description="Phần trăm thời gian im lặng (%)")
    babbling_detected: bool = Field(..., description="Có phát hiện bập bẹ không")
    vocalizations: List[Dict[str, Any]] = Field(..., description="Danh sách các vocalization events với timestamp")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100, cao hơn = rủi ro cao hơn)")
    # Thông tin phân loại giọng nói
    child_speech_segments: List[Dict[str, Any]] = Field(default=[], description="Các segments được phân loại là giọng trẻ em")
    adult_speech_segments: List[Dict[str, Any]] = Field(default=[], description="Các segments được phân loại là giọng người lớn")
    child_speech_duration: float = Field(default=0.0, description="Thời lượng giọng trẻ em (giây)")
    adult_speech_duration: float = Field(default=0.0, description="Thời lượng giọng người lớn (giây)")
    child_speech_percentage: float = Field(default=0.0, description="Phần trăm thời gian giọng trẻ em (%)")
    adult_speech_percentage: float = Field(default=0.0, description="Phần trăm thời gian giọng người lớn (%)")


@router.post("/analyze", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    file: UploadFile = File(..., description="Audio hoặc Video file để phân tích (wav, mp3, mp4, etc.)"),
    show_video: str = Form("true", description="Hiển thị video với annotations (true/false)")
):
    """
    Phân tích Speech / Audio từ file audio hoặc video
    
    - Nhận tần suất âm thanh
    - Phát hiện ít nói, bập bẹ
    - Nếu là video, sẽ extract audio và hiển thị kết quả trên video
    - Không cần ASR phức tạp, chỉ cần detect vocalization
    
    Args:
        file: File audio (wav, mp3, m4a) hoặc video (mp4, avi, mov, etc.)
        show_video: "true" để hiển thị video với annotations
    
    Returns:
        SpeechAnalysisResponse với các chỉ số phân tích
    """
    temp_path = None
    temp_audio_path = None
    temp_video_path = None
    is_video = False
    show_video_bool = show_video.lower() in ("true", "1", "yes", "on")
    
    try:
        # Lưu file tạm
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Kiểm tra xem là video hay audio
        file_ext = os.path.splitext(file.filename)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        is_video = file_ext in video_extensions
        
        if is_video:
            logger.info(f"[Speech] File là video ({file_ext}), sẽ extract audio")
            # Extract audio từ video
            temp_audio_path = f"temp_audio_{file.filename}.wav"
            extract_audio_from_video(temp_path, temp_audio_path)
            audio_path = temp_audio_path
            temp_video_path = temp_path
        else:
            logger.info(f"[Speech] File là audio ({file_ext})")
            audio_path = temp_path
        
        if not LIBROSA_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="librosa không được cài đặt. Vui lòng cài: pip install librosa soundfile"
            )
        
        # Load audio file
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            audio_duration = len(y) / sr
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Không thể load audio file: {str(e)}"
            )
        
        # Voice Activity Detection
        voice_activity, speech_duration, rms_energy = detect_voice_activity(y, sr)
        speech_percentage = (speech_duration / audio_duration * 100) if audio_duration > 0 else 0
        silence_percentage = 100 - speech_percentage
        
        # Detect vocalizations
        vocalizations = detect_vocalizations(y, sr)
        vocalization_count = len(vocalizations)
        vocalization_frequency = vocalization_count / audio_duration if audio_duration > 0 else 0
        
        # Detect babbling
        babbling_detected, voice_percentage, variation = detect_babbling(y, sr)
        
        # Phân loại giọng nói: trẻ em vs người lớn
        logger.info("[Speech] Đang phân loại giọng nói (trẻ em vs người lớn)...")
        speech_segments = detect_speakers_and_classify(y, sr, voice_activity)
        
        # Phân loại segments
        child_segments = [s for s in speech_segments if s['voice_type'] == 'child']
        adult_segments = [s for s in speech_segments if s['voice_type'] == 'adult']
        
        # Tính thời lượng
        child_speech_duration = sum(s['duration'] for s in child_segments)
        adult_speech_duration = sum(s['duration'] for s in adult_segments)
        
        # Tính phần trăm
        child_speech_percentage = (child_speech_duration / audio_duration * 100) if audio_duration > 0 else 0
        adult_speech_percentage = (adult_speech_duration / audio_duration * 100) if audio_duration > 0 else 0
        
        logger.info(f"[Speech] Phân loại hoàn tất:")
        logger.info(f"  - Giọng trẻ em: {len(child_segments)} segments, {child_speech_duration:.2f}s ({child_speech_percentage:.1f}%)")
        logger.info(f"  - Giọng người lớn: {len(adult_segments)} segments, {adult_speech_duration:.2f}s ({adult_speech_percentage:.1f}%)")
        
        # Tính risk score (ít nói = risk cao)
        # Trẻ ASD thường ít nói, ít bập bẹ
        base_risk = max(0, min(100, (100 - speech_percentage) * 0.7))
        
        # Giảm risk nếu có babbling
        if babbling_detected:
            base_risk = base_risk * 0.8
        
        # Tăng risk nếu vocalization frequency quá thấp
        if vocalization_frequency < 0.5:  # Ít hơn 0.5 vocalizations/giây
            base_risk = min(100, base_risk + 15)
        
        # Điều chỉnh risk dựa trên child speech percentage
        # Nếu trẻ nói quá ít so với người lớn, có thể là dấu hiệu ASD
        if child_speech_percentage > 0 and adult_speech_percentage > 0:
            child_adult_ratio = child_speech_percentage / adult_speech_percentage
            if child_adult_ratio < 0.3:  # Trẻ nói ít hơn 30% so với người lớn
                base_risk = min(100, base_risk + 10)
        
        risk_score = base_risk
        
        # Nếu là video, hiển thị kết quả trên video
        if is_video and show_video_bool:
            cap = None
            try:
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    logger.warning("[Speech] Không thể mở video để hiển thị")
                else:
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    video_fps = video_fps if video_fps > 0 else 30
                    
                    # Map voice activity từ audio time sang video frames
                    frames_per_second_audio = sr / 512
                    
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Tính thời gian hiện tại trong video
                        current_time = frame_count / video_fps
                        
                        # Tìm voice activity tại thời điểm này
                        audio_frame_idx = int(current_time * frames_per_second_audio)
                        voice_activity_frame = False
                        if 0 <= audio_frame_idx < len(voice_activity):
                            voice_activity_frame = voice_activity[audio_frame_idx]
                        
                        # Tìm vocalizations tại thời điểm này
                        vocalizations_at_time = [
                            v for v in vocalizations
                            if v['start_time'] <= current_time <= v['end_time']
                        ]
                        
                        # Tìm speech segments tại thời điểm này
                        current_child_segments = [
                            s for s in child_segments
                            if s['start_time'] <= current_time <= s['end_time']
                        ]
                        current_adult_segments = [
                            s for s in adult_segments
                            if s['start_time'] <= current_time <= s['end_time']
                        ]
                        
                        # Vẽ annotations
                        annotated_frame = draw_speech_annotations_on_video(
                            frame, current_time, voice_activity_frame,
                            vocalizations_at_time, babbling_detected,
                            speech_percentage, frame_count, video_fps,
                            child_segments=child_segments,
                            adult_segments=adult_segments,
                            child_speech_percentage=child_speech_percentage,
                            adult_speech_percentage=adult_speech_percentage
                        )
                        
                        try:
                            # Resize nếu frame quá lớn
                            h, w = frame.shape[:2]
                            display_frame = annotated_frame.copy()
                            max_width = 1280
                            if w > max_width:
                                scale = max_width / w
                                new_width = max_width
                                new_height = int(h * scale)
                                display_frame = cv2.resize(display_frame, (new_width, new_height))
                            
                            cv2.imshow("Speech Analysis - Press 'q' to quit", display_frame)
                            key = cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF
                            if key == ord('q'):
                                logger.info("[Speech] User pressed 'q', stopping video display")
                                show_video_bool = False
                        except cv2.error as e:
                            if "No display" in str(e) or "cannot connect" in str(e).lower():
                                logger.warning("[Speech] Không thể hiển thị video (headless server).")
                                show_video_bool = False
                            else:
                                raise
                        except Exception as e:
                            logger.warning(f"[Speech] Lỗi khi hiển thị video: {str(e)}")
                        
                        frame_count += 1
                    
                    if cap:
                        cap.release()
                    
                    if show_video_bool:
                        cv2.destroyAllWindows()
            except Exception as e:
                logger.warning(f"[Speech] Không thể hiển thị video: {str(e)}")
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
        
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        
        # Format vocalizations với timestamp
        vocalizations_formatted = [
            {
                'start_time': round(v['start_time'], 2),
                'end_time': round(v['end_time'], 2),
                'duration': round(v['duration'], 2)
            }
            for v in vocalizations
        ]
        
        # Format speech segments
        child_segments_formatted = [
            {
                'start_time': round(s['start_time'], 2),
                'end_time': round(s['end_time'], 2),
                'duration': round(s['duration'], 2),
                'pitch_mean': round(s['pitch_mean'], 2) if s['pitch_mean'] else None,
                'pitch_median': round(s['pitch_median'], 2) if s['pitch_median'] else None,
                'formants': [round(f, 2) for f in s['formants']] if s['formants'] else None
            }
            for s in child_segments
        ]
        
        adult_segments_formatted = [
            {
                'start_time': round(s['start_time'], 2),
                'end_time': round(s['end_time'], 2),
                'duration': round(s['duration'], 2),
                'pitch_mean': round(s['pitch_mean'], 2) if s['pitch_mean'] else None,
                'pitch_median': round(s['pitch_median'], 2) if s['pitch_median'] else None,
                'formants': [round(f, 2) for f in s['formants']] if s['formants'] else None
            }
            for s in adult_segments
        ]
        
        return SpeechAnalysisResponse(
            audio_duration=round(audio_duration, 2),
            speech_duration=round(speech_duration, 2),
            speech_percentage=round(speech_percentage, 2),
            vocalization_frequency=round(vocalization_frequency, 2),
            silence_percentage=round(silence_percentage, 2),
            babbling_detected=babbling_detected,
            vocalizations=vocalizations_formatted,
            risk_score=round(risk_score, 2),
            child_speech_segments=child_segments_formatted,
            adult_speech_segments=adult_segments_formatted,
            child_speech_duration=round(child_speech_duration, 2),
            adult_speech_duration=round(adult_speech_duration, 2),
            child_speech_percentage=round(child_speech_percentage, 2),
            adult_speech_percentage=round(adult_speech_percentage, 2)
        )
        
    except HTTPException:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        raise
    except Exception as e:
        logger.error(f"[Speech] Lỗi khi phân tích audio: {str(e)}")
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích speech: {str(e)}"
        )

