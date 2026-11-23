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

        # Define command groups
        self.command_groups = {
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
                for command_group in self.command_groups.values():
                    if lang in command_group:
                        for word in command_group[lang]:
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