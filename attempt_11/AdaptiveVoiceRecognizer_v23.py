import speech_recognition as sr
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import pygame
import os
from difflib import SequenceMatcher
import json
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
import re


class AdaptiveVoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True  # Adjust for changing noise levels

        # Try to initialize microphone with error handling
        self.microphone_available = False
        self.microphone = None

        try:
            self.microphone = sr.Microphone()
            # Test if microphone is actually accessible
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.microphone_available = True
            print("Microphone initialized successfully")
        except OSError as e:
            print(f"Microphone initialization error: {e}")
        except Exception as e:
            print(f"Error initializing microphone: {e}")

        self.command_mappings = {
            "next": {
                "en": ["next", "next one", "right", "forward"],
                "es": ["siguiente", "derecha", "adelante"],
                "fr": ["suivant", "au suivant", "droite"],
                "ar": ["التالي", "يمين"],
                "it": ["successivo", "destra"],
                "de": ["nächste", "rechts"],
                "pt": ["próximo", "direita"]
            },
            "previous": {
                "en": ["previous", "previous one", "back", "left", "go back"],
                "es": ["anterior", "atrás", "izquierda", "retroceder"],
                "fr": ["précédent", "au précédent", "gauche"],
                "ar": ["السابق", "خلف", "يسار"],
                "it": ["precedente", "indietro", "sinistra"],
                "de": ["vorherige", "zurück", "links"],
                "pt": ["anterior", "voltar", "esquerda"]
            },
            "exit": {
                "en": ["exit", "quit", "close", "end", "stop"],
                "es": ["salir", "terminar", "cerrar", "finalizar"],
                "fr": ["quitter", "sortir", "fermer", "arrêter"],
                "ar": ["خروج", "إنهاء", "إغلاق", "توقف"],
                "it": ["uscire", "chiudere", "terminare", "fermare"],
                "de": ["beenden", "verlassen", "schließen", "stoppen"],
                "pt": ["sair", "fechar", "terminar", "parar"]
            },
            "go to": {
                "en": ["go to", "navigate to", "jump to", "select", "choose"],
                "es": ["ir a", "navegar a", "saltar a", "seleccionar", "elegir"],
                "fr": ["aller à", "naviguer vers", "sauter à", "sélectionner", "choisir"],
                "ar": ["اذهب إلى", "انتقل إلى", "اختر", "حدد"],
                "it": ["vai a", "naviga verso", "salta a", "seleziona", "scegli"],
                "de": ["gehe zu", "navigiere zu", "springe zu", "wähle", "auswählen"],
                "pt": ["ir para", "navegar para", "pular para", "selecionar", "escolher"]
            },
            "first": {
                "en": ["first", "beginning", "start"],
                "es": ["primero", "inicio", "comienzo"],
                "fr": ["premier", "début", "commencement"],
                "ar": ["أول", "بداية", "البداية"],
                "it": ["primo", "inizio", "iniziare"],
                "de": ["erste", "anfang", "beginn"],
                "pt": ["primeiro", "início", "começo"]
            },
            "last": {
                "en": ["last", "end", "final"],
                "es": ["último", "final", "fin"],
                "fr": ["dernier", "fin", "final"],
                "ar": ["آخر", "نهاية", "الختام"],
                "it": ["ultimo", "fine", "finale"],
                "de": ["letzte", "ende", "finale"],
                "pt": ["último", "fim", "final"]
            }
        }

        # Language codes for speech recognition
        self.lang_codes = {
            "en": "en-US",  # English
            "es": "es-ES",  # Spanish
            "fr": "fr-FR",  # French
            "ar": "ar",  # Arabic
            "it": "it-IT",  # Italian
            "de": "de-DE",  # German
            "pt": "pt-PT"  # Portuguese
        }

        self.active_words = set()
        self.selected_languages = []
        self.is_listening = False
        self.is_testing_mic = False

        # Adaptive learning components
        self.user_profile = {}
        self.similarity_threshold = 0.6  # Initial threshold for fuzzy matching
        self.learning_rate = 0.05  # How quickly the system adapts
        self.profile_file = "user_pronunciation_profile.pkl"
        self.load_user_profile()

        # Initialize pygame for sounds
        pygame.mixer.init()
        self.sound_start = None
        self.sound_success = None
        self.sound_failure = None

        # Create sounds
        self.create_sounds()

        # Adjust for ambient noise
        self.adjust_ambient_noise()

    def load_user_profile(self):
        """Load user pronunciation profile if it exists"""
        try:
            if Path(self.profile_file).exists():
                with open(self.profile_file, 'rb') as f:
                    self.user_profile = pickle.load(f)
                print("Loaded user pronunciation profile")
        except Exception as e:
            print(f"Error loading user profile: {e}")
            self.user_profile = {}

    def save_user_profile(self):
        """Save user pronunciation profile"""
        try:
            with open(self.profile_file, 'wb') as f:
                pickle.dump(self.user_profile, f)
        except Exception as e:
            print(f"Error saving user profile: {e}")

    def calculate_similarity(self, word1, word2):
        """Calculate similarity between two words (0 to 1)"""
        return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

    def find_best_match(self, spoken_word, target_words):
        """Find the best match for the spoken word among target words"""
        best_match = None
        highest_similarity = 0

        for target_word in target_words:
            # Check if we have user-specific similarity data
            user_similarity_key = f"{spoken_word}_{target_word}"
            if user_similarity_key in self.user_profile:
                similarity = self.user_profile[user_similarity_key]
            else:
                similarity = self.calculate_similarity(spoken_word, target_word)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = target_word

        return best_match, highest_similarity

    def update_user_profile(self, spoken_word, correct_word, similarity):
        """Update the user profile with pronunciation data"""
        user_similarity_key = f"{spoken_word}_{correct_word}"

        # Update or set the similarity score
        if user_similarity_key in self.user_profile:
            # Moving average to adapt gradually
            self.user_profile[user_similarity_key] = (
                    (1 - self.learning_rate) * self.user_profile[user_similarity_key] +
                    self.learning_rate * similarity
            )
        else:
            self.user_profile[user_similarity_key] = similarity

        # Adjust the global similarity threshold based on user performance
        if similarity > self.similarity_threshold:
            # User is getting better, we can be slightly more strict
            self.similarity_threshold = min(0.8, self.similarity_threshold + 0.01)
        else:
            # User needs more flexibility
            self.similarity_threshold = max(0.4, self.similarity_threshold - 0.01)

        self.save_user_profile()

    def create_sounds(self):
        # Create simple beep sounds using pygame
        try:
            # Start sound (high beep)
            self.sound_start = pygame.mixer.Sound(buffer=bytearray([128] * 44100))
            self.sound_start.set_volume(0.5)

            # Success sound (rising tone)
            success_data = bytearray()
            for i in range(22050):
                value = int(127 + 127 * (i / 22050) * 0.5)
                success_data.append(value)
            self.sound_success = pygame.mixer.Sound(buffer=success_data)
            self.sound_success.set_volume(0.5)

            # Failure sound (falling tone)
            failure_data = bytearray()
            for i in range(22050):
                value = int(127 + 127 * (1 - i / 22050) * 0.5)
                failure_data.append(value)
            self.sound_failure = pygame.mixer.Sound(buffer=failure_data)
            self.sound_failure.set_volume(0.5)
        except:
            print("Could not create sounds, continuing without audio feedback")

    def adjust_ambient_noise(self):
        def adjust():
            try:
                if self.microphone_available:
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print(f"Error adjusting for ambient noise: {e}")

        thread = threading.Thread(target=adjust)
        thread.daemon = True
        thread.start()

    def set_target_words(self, languages):
        """Set target words based on selected languages"""
        self.active_words.clear()
        self.selected_languages = languages

        # Add all words from selected languages
        for lang in languages:
            if lang in ["en", "es", "fr", "ar", "it", "de", "pt"]:
                for command_mappings in self.command_mappings.values():
                    if lang in command_mappings:
                        for word in command_mappings[lang]:
                            self.active_words.add(word.lower())

        return f"Listening for commands in: {', '.join(languages)}"

    def recognize_speech(self, audio):
        """Try to recognize speech in all selected languages and return the best match"""
        best_match = None
        best_confidence = 0
        recognized_text = ""

        # Try recognition in each selected language
        for lang in self.selected_languages:
            try:
                lang_code = self.lang_codes.get(lang, "en-US")
                text = self.recognizer.recognize_google(audio, language=lang_code).lower()
                recognized_text = text

                # Use fuzzy matching to find the best command match
                best_command, similarity = self.find_best_match(text, self.active_words)

                if best_command and similarity >= self.similarity_threshold:
                    return text, best_command, similarity

            except sr.UnknownValueError:
                # Could not understand audio in this language, try next
                continue
            except Exception as e:
                # Other errors, continue with next language
                continue

        return recognized_text, None, 0

    def listen_continuously(self, status_callback, command_callback):
        """Continuously listen for voice commands"""
        if not self.microphone_available:
            status_callback("Microphone not available")
            return

        self.is_listening = True
        status_callback("Listening...")

        while self.is_listening:
            try:
                with self.microphone as source:
                    if self.sound_start:
                        self.sound_start.play()

                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

                    # Recognize speech
                    text, command, confidence = self.recognize_speech(audio)

                    if command:
                        if self.sound_success:
                            self.sound_success.play()

                        # Process the command
                        success = self.process_command(command, text)

                        if success:
                            status_callback(f"Executed: {command}")
                            command_callback(text, command, confidence, True)
                        else:
                            status_callback(f"Failed to execute: {command}")
                            command_callback(text, command, confidence, False)
                    else:
                        if self.sound_failure:
                            self.sound_failure.play()
                        status_callback(f"Could not understand: {text}")
                        command_callback(text, None, 0, False)

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except Exception as e:
                status_callback(f"Error: {str(e)}")
                time.sleep(1)  # Prevent tight loop on error

        status_callback("Stopped listening")

    def stop_listening(self):
        """Stop the continuous listening loop"""
        self.is_listening = False

    def set_target_words(self, languages):
        """Set target words based on selected languages"""
        self.active_words.clear()
        self.selected_languages = languages

        # Add all words from selected languages
        for lang in languages:
            if lang in ["en", "es", "fr", "ar", "it", "de", "pt"]:
                for command_mappings in self.command_mappings.values():
                    if lang in command_mappings:
                        for word in command_mappings[lang]:
                            self.active_words.add(word.lower())

        return f"Listening for commands in: {', '.join(languages)}"

    def recognize_speech(self, audio):
        """Try to recognize speech in all selected languages and return the best match"""
        best_match = None
        best_confidence = 0
        recognized_text = ""

        # Try recognition in each selected language
        for lang in self.selected_languages:
            try:
                lang_code = self.lang_codes.get(lang, "en-US")
                text = self.recognizer.recognize_google(audio, language=lang_code).lower()
                recognized_text = text

                # Use fuzzy matching to find the best command match
                best_command, similarity = self.find_best_match(text, self.active_words)

                if best_command and similarity >= self.similarity_threshold:
                    return text, best_command, similarity

            except sr.UnknownValueError:
                # Could not understand audio in this language, try next
                continue
            except Exception as e:
                # Other errors, continue with next language
                continue

        return recognized_text, None, 0

    def update_user_profile(self, spoken_word, correct_word, similarity):
        """Update the user profile with pronunciation data"""
        user_similarity_key = f"{spoken_word}_{correct_word}"

        # Update or set the similarity score
        if user_similarity_key in self.user_profile:
            # Moving average to adapt gradually
            self.user_profile[user_similarity_key] = (
                    (1 - self.learning_rate) * self.user_profile[user_similarity_key] +
                    self.learning_rate * similarity
            )
        else:
            self.user_profile[user_similarity_key] = similarity

        # Adjust the global similarity threshold based on user performance
        if similarity > self.similarity_threshold:
            # User is getting better, we can be slightly more strict
            self.similarity_threshold = min(0.8, self.similarity_threshold + 0.01)
        else:
            # User needs more flexibility
            self.similarity_threshold = max(0.4, self.similarity_threshold - 0.01)

        self.save_user_profile()

    def extract_slide_number(self, text):
        """Extract slide number from voice command"""
        # First, try to extract numbers using regex
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])

        # Look for number words in all supported languages
        number_words = {
            # English
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            # Spanish
            'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
            'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
            # French
            'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
            'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
            # Arabic (transliterated)
            'wahid': 1, 'ithnan': 2, 'thalatha': 3, 'arba\'a': 4, 'khamsa': 5,
            'sitta': 6, 'sab\'a': 7, 'thamaniya': 8, 'tis\'a': 9, 'ashara': 10,
            # Italian
            'uno': 1, 'due': 2, 'tre': 3, 'quattro': 4, 'cinque': 5,
            'sei': 6, 'sette': 7, 'otto': 8, 'nove': 9, 'dieci': 10,
            # German
            'eins': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
            'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
            # Portuguese
            'um': 1, 'dois': 2, 'três': 3, 'quatro': 4, 'cinco': 5,
            'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10
        }

        # Split text into words and check each word
        words = text.lower().split()
        for word in words:
            if word in number_words:
                return number_words[word]

        # Also check for combined phrases like "go to five"
        for number_word, number in number_words.items():
            if number_word in text.lower():
                return number

        return None
