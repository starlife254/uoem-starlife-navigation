"""
Voice Processing Module for AI Campus Navigation
Supports speech recognition and voice commands
"""

import speech_recognition as sr
import threading
import queue
import json
import logging
from typing import Optional, Callable, Dict
import time

# Try to import pyaudio, but don't fail if it's not available
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
    print("✅ PyAudio available - microphone support enabled")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠ PyAudio not installed - microphone recording disabled")
    # Create dummy classes/functions
    class pyaudio:
        paInt16 = 8  # Dummy constant
        def __init__(self): pass
        def open(self, *args, **kwargs): return None
        def terminate(self): pass
    
    class wave:
        @staticmethod
        def open(*args, **kwargs): return None

class VoiceProcessor:
    """Handle voice input and speech recognition"""
    
    def __init__(self, language: str = 'en-US', energy_threshold: int = 300):
        """
        Initialize voice processor
        
        Args:
            language: Speech recognition language (en-US, sw-KE)
            energy_threshold: Energy threshold for speech detection
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.energy_threshold = energy_threshold
        self.is_listening = False
        self.recognition_thread = None
        self.audio_queue = queue.Queue()
        self.callback = None
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        
        # Check if microphone is available
        self.microphone_available = False
        try:
            if PYAUDIO_AVAILABLE:
                self.microphone = sr.Microphone()
                self.microphone_available = True
                print("✅ Microphone available")
            else:
                self.microphone = None
                print("⚠ Microphone not available (PyAudio missing)")
        except Exception as e:
            self.microphone = None
            print(f"⚠ Microphone initialization failed: {e}")
        
        # Supported languages
        self.supported_languages = {
            'english': 'en-US',
            'swahili': 'sw-KE',
            'en': 'en-US',
            'sw': 'sw-KE'
        }
        
        # Voice commands database
        self.voice_commands = self._load_voice_commands()
        
        print(f"✅ Voice Processor initialized for language: {language}")
        if not self.microphone_available:
            print("   Note: Voice input will work with uploaded files only")
    
    def _load_voice_commands(self) -> Dict:
        """Load predefined voice commands"""
        return {
            'navigation': [
                "where is", "how do i get to", "directions to", "route to",
                "go to", "take me to", "navigate to", "find", "show me",
                "iko wapi", "naenda wapi", "nifikie wapi", "elekea"
            ],
            'control': [
                "start navigation", "stop navigation", "clear route",
                "show buildings", "hide buildings", "my location",
                "anza uelekezaji", "acha uelekezaji", "futa njia"
            ],
            'information': [
                "what is", "tell me about", "information about",
                "ni nini", "elezea kuhusu", "habari za"
            ]
        }
    
    def set_language(self, language: str):
        """Set recognition language"""
        if language in self.supported_languages:
            self.language = self.supported_languages[language]
            print(f"🌐 Language set to: {language}")
        else:
            print(f"⚠ Language {language} not supported, using English")
            self.language = 'en-US'
    
    def start_listening(self, callback: Callable[[str], None] = None):
        """
        Start listening for voice input
        
        Args:
            callback: Function to call when speech is recognized
        """
        if not self.microphone_available:
            print("⚠ Cannot start listening: Microphone not available")
            if callback:
                callback("Microphone not available. Please use text input.")
            return
        
        if self.is_listening:
            print("⚠ Already listening")
            return
        
        self.callback = callback
        self.is_listening = True
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self._recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        print("🎤 Started listening for voice input...")
    
    def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2)
        print("🎤 Stopped listening")
    
    def _recognition_loop(self):
        """Main recognition loop"""
        if not self.microphone_available:
            return
            
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                while self.is_listening:
                    try:
                        print("👂 Listening... (speak now)")
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        
                        # Recognize speech
                        text = self.recognizer.recognize_google(audio, language=self.language)
                        
                        if text:
                            print(f"🗣 Recognized: {text}")
                            
                            # Process command
                            processed_text = self._process_voice_command(text)
                            
                            # Call callback if provided
                            if self.callback:
                                self.callback(processed_text)
                            
                            # Add to queue
                            self.audio_queue.put({
                                'text': text,
                                'processed': processed_text,
                                'timestamp': time.time()
                            })
                            
                    except sr.WaitTimeoutError:
                        # No speech detected, continue listening
                        continue
                    except sr.UnknownValueError:
                        print("⚠ Could not understand audio")
                    except sr.RequestError as e:
                        print(f"⚠ Recognition error: {e}")
                    except Exception as e:
                        print(f"⚠ Error in recognition loop: {e}")
                        
        except Exception as e:
            print(f"⚠ Error setting up microphone: {e}")
            self.is_listening = False
    
    def _process_voice_command(self, text: str) -> str:
        """Process and normalize voice command"""
        text_lower = text.lower()
        
        # Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically']
        for word in filler_words:
            text_lower = text_lower.replace(word, '')
        
        # Normalize common phrases
        replacements = {
            "i'd like to go to": "go to",
            "i want to go to": "go to",
            "can you take me to": "take me to",
            "please show me": "show me",
            "could you find": "find",
            "nataka kwenda": "naenda",
            "naomba uniongoze": "elekea"
        }
        
        for old, new in replacements.items():
            if old in text_lower:
                text_lower = text_lower.replace(old, new)
        
        return text_lower.strip()
    
    def recognize_from_file(self, audio_file: str) -> Optional[str]:
        """
        Recognize speech from audio file
        
        Args:
            audio_file: Path to audio file (WAV format)
        
        Returns:
            Recognized text or None
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
        except Exception as e:
            print(f"⚠ Error recognizing from file: {e}")
            return None
    
    def record_audio(self, duration: int = 5, output_file: str = "recording.wav"):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            output_file: Output WAV file path
        """
        if not PYAUDIO_AVAILABLE:
            print("⚠ Cannot record audio: PyAudio not installed")
            return
            
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1,
                               rate=44100, input=True, frames_per_buffer=1024)
            
            print(f"🎤 Recording for {duration} seconds...")
            frames = []
            
            for _ in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            print("✅ Recording complete")
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save to file
            import wave
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))
            
            print(f"💾 Saved recording to {output_file}")
            
        except Exception as e:
            print(f"⚠ Error recording audio: {e}")
    
    def get_next_command(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next recognized command from queue
        
        Args:
            timeout: Queue timeout in seconds
        
        Returns:
            Command dictionary or None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self):
        """Clear the audio queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_supported_commands(self) -> Dict:
        """Get supported voice commands by category"""
        return self.voice_commands
    
    def is_command_type(self, text: str, command_type: str) -> bool:
        """
        Check if text matches a command type
        
        Args:
            text: Recognized text
            command_type: Type of command to check
        
        Returns:
            True if text matches command type
        """
        if command_type not in self.voice_commands:
            return False
        
        text_lower = text.lower()
        for command in self.voice_commands[command_type]:
            if command in text_lower:
                return True
        
        return False
    
    def extract_location_from_command(self, text: str) -> Optional[str]:
        """
        Extract location/building from voice command
        
        Args:
            text: Recognized text
        
        Returns:
            Extracted location or None
        """
        text_lower = text.lower()
        
        # Remove command phrases
        for command_type in self.voice_commands.values():
            for command in command_type:
                if command in text_lower:
                    text_lower = text_lower.replace(command, '').strip()
                    break
        
        # Clean up
        text_lower = text_lower.strip(' to').strip()
        
        return text_lower if text_lower else None

# Singleton instance
_voice_processor_instance = None

def get_voice_processor(language: str = 'en-US') -> VoiceProcessor:
    """Get or create voice processor instance"""
    global _voice_processor_instance
    
    if _voice_processor_instance is None:
        _voice_processor_instance = VoiceProcessor(language)
    
    return _voice_processor_instance

def voice_to_text(audio_file: str = None, language: str = 'en-US') -> Optional[str]:
    """
    Convert voice to text (simple interface)
    
    Args:
        audio_file: Path to audio file (if None, uses microphone)
        language: Recognition language
    
    Returns:
        Recognized text or None
    """
    processor = get_voice_processor(language)
    
    if audio_file:
        return processor.recognize_from_file(audio_file)
    
    # Use microphone (will fail gracefully if not available)
    result = None
    
    def callback(text):
        nonlocal result
        result = text
    
    processor.start_listening(callback)
    time.sleep(3)  # Listen for 3 seconds
    processor.stop_listening()
    
    return result