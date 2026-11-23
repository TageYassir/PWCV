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
import pyautogui
import re


class AdaptiveVoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        # Try to initialize microphone with error handling
        self.microphone_available = False
        self.microphone = None

        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.microphone_available = True
            print("Microphone initialized successfully")
        except OSError as e:
            print(f"Microphone initialization error: {e}")
        except Exception as e:
            print(f"Error initializing microphone: {e}")


        # Language codes for speech recognition
        self.lang_codes = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "ar": "ar",
            "it": "it-IT",
            "de": "de-DE",
            "pt": "pt-PT"
        }

        self.active_words = set()
        self.selected_languages = []
        self.is_listening = False
        self.is_testing_mic = False

        # Adaptive learning components - maintaining the same logic
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
                    # Update user profile with this recognition
                    self.update_user_profile(text, best_command, similarity)
                    return text, best_command, similarity

            except sr.UnknownValueError:
                continue
            except Exception as e:
                continue

        return recognized_text, None, 0

    def extract_slide_number(self, text):
        """Extract slide number from voice command"""
        # Look for numbers in the text
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])

        # Look for number words (only English for simplicity)
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }

        for word, number in number_words.items():
            if word in text:
                return number

        return None

    def navigate_to_slide(self, slide_number):
        """Navigate to the specified slide number using PowerPoint's Ctrl+S shortcut"""
        try:
            # Press Ctrl+S to open the "All Slides" dialog
            pyautogui.hotkey('ctrl', 's')
            time.sleep(0.5)  # Wait for the dialog to appear

            # Type the slide number
            pyautogui.write(str(slide_number))
            time.sleep(0.2)

            # Press Enter to navigate to the slide
            pyautogui.press('enter')

            return True
        except Exception as e:
            print(f"Error navigating to slide: {e}")
            return False

    def process_command(self, command, text):
        """Process the recognized command"""
        # Define command mappings for all languages
        command_mappings = {
            "next": {
                "en": ["next", "next one", "right", "forward", "next slide"],
                "es": ["siguiente", "derecha", "adelante"],
                "fr": ["suivant", "au suivant", "droite"],
                "ar": ["التالي", "يمين"],
                "it": ["successivo", "destra"],
                "de": ["nächste", "rechts"],
                "pt": ["próximo", "direita"]
            },
            "previous": {
                "en": ["previous", "previous one", "back", "left", "go back", "previous slide"],
                "es": ["anterior", "atrás", "izquierda", "retroceder"],
                "fr": ["précédent", "au précédent", "gauche"],
                "ar": ["السابق", "خلف", "يسار"],
                "it": ["precedente", "indietro", "sinistra"],
                "de": ["vorherige", "zurück", "links"],
                "pt": ["anterior", "voltar", "esquerda"]
            },
            "exit": {
                "en": ["exit", "quit", "close", "end", "stop presentation", "exit presentation"],
                "es": ["salir", "terminar", "cerrar", "finalizar"],
                "fr": ["quitter", "sortir", "fermer", "arrêter"],
                "ar": ["خرج", "إنهاء", "إغلاق"],
                "it": ["uscire", "chiudere", "terminare"],
                "de": ["beenden", "verlassen", "schließen"],
                "pt": ["sair", "fechar", "terminar"]
            },
            "first": {
                "en": ["first", "first slide", "beginning", "start"],
                "es": ["primero", "primera diapositiva", "inicio", "comienzo"],
                "fr": ["premier", "première diapositive", "début"],
                "ar": ["أول", "الشريحة الأولى", "بداية"],
                "it": ["primo", "prima diapositiva", "inizio"],
                "de": ["erste", "erste folie", "anfang"],
                "pt": ["primeiro", "primeiro slide", "início"]
            },
            "last": {
                "en": ["last", "last slide", "end", "final"],
                "es": ["último", "última diapositiva", "final"],
                "fr": ["dernier", "dernière diapositive", "fin"],
                "ar": ["آخر", "الشريحة الأخيرة", "نهاية"],
                "it": ["ultimo", "ultima diapositiva", "fine"],
                "de": ["letzte", "letzte folie", "ende"],
                "pt": ["último", "último slide", "fim"]
            },
            "goto": {
                "en": ["go to", "jump to", "navigate to", "slide"],
                "es": ["ir a", "saltar a", "navegar a", "diapositiva"],
                "fr": ["aller à", "sauter à", "naviguer à", "diapositive"],
                "ar": ["اذهب إلى", "انتقل إلى", "شريحة"],
                "it": ["vai a", "salta a", "naviga a", "diapositiva"],
                "de": ["gehe zu", "springe zu", "navigiere zu", "folie"],
                "pt": ["ir para", "pular para", "navegar para", "slide"]
            }
        }

        # Check if the command matches any of the multilingual options
        normalized_text = text.lower().strip()

        for action, languages in command_mappings.items():
            for lang, phrases in languages.items():
                if normalized_text in phrases:
                    command = action
                    break

        # Handle the command
        if command == "next" or any(normalized_text in phrases for phrases in command_mappings["next"].values()):
            pyautogui.press('right')
        elif command == "previous" or any(
                normalized_text in phrases for phrases in command_mappings["previous"].values()):
            pyautogui.press('left')
        elif command == "first" or any(normalized_text in phrases for phrases in command_mappings["first"].values()):
            pyautogui.hotkey('ctrl', 'shift', 'home')
        elif command == "last" or any(normalized_text in phrases for phrases in command_mappings["last"].values()):
            pyautogui.hotkey('ctrl', 'shift', 'end')
        elif command == "exit" or any(normalized_text in phrases for phrases in command_mappings["exit"].values()):
            pyautogui.press('esc')
        elif command == "goto" or any(normalized_text in phrases for phrases in command_mappings["goto"].values()):
            # Extract slide number from text using regex
            numbers = re.findall(r'\d+', text)
            if numbers:
                slide_number = int(numbers[0])
                # Implementation for going to a specific slide would depend on your presentation software
                # This is a generic implementation that might need adjustment
                pyautogui.hotkey('ctrl', 'g')  # Common shortcut for "Go To" in presentation software
                pyautogui.write(str(slide_number))
                pyautogui.press('enter')

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


class VoiceControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PowerPoint Voice Navigation")
        self.root.geometry("600x500")

        self.recognizer = AdaptiveVoiceRecognizer()

        # Create GUI
        self.create_widgets()


    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="PowerPoint Voice Navigation", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Language selection
        lang_frame = ttk.LabelFrame(main_frame, text="Language Settings", padding="5")
        lang_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.lang_vars = {}
        languages = ["en", "es", "fr", "ar", "it", "de", "pt"]
        lang_names = ["English", "Spanish", "French", "Arabic", "Italian", "German", "Portuguese"]

        for i, (code, name) in enumerate(zip(languages, lang_names)):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(lang_frame, text=name, variable=var,
                                  command=self.update_languages)
            chk.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)
            self.lang_vars[code] = var

        # Set English as default
        self.lang_vars["en"].set(True)

        # Status label
        self.status_var = tk.StringVar(value="Ready to listen")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Command log
        log_frame = ttk.LabelFrame(main_frame, text="Command Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Create a treeview for the log
        columns = ("time", "text", "command", "confidence", "success")
        self.log_tree = ttk.Treeview(log_frame, columns=columns, show="headings", height=10)

        # Define headings
        self.log_tree.heading("time", text="Time")
        self.log_tree.heading("text", text="Recognized Text")
        self.log_tree.heading("command", text="Command")
        self.log_tree.heading("confidence", text="Confidence")
        self.log_tree.heading("success", text="Success")

        # Define columns
        self.log_tree.column("time", width=80)
        self.log_tree.column("text", width=150)
        self.log_tree.column("command", width=100)
        self.log_tree.column("confidence", width=80)
        self.log_tree.column("success", width=60)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=scrollbar.set)

        self.log_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        # Start/Stop buttons
        self.start_button = ttk.Button(button_frame, text="Start Listening", command=self.start_listening)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Listening", command=self.stop_listening,
                                      state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Instructions
        instr_frame = ttk.LabelFrame(main_frame, text="Voice Commands", padding="5")
        instr_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        instr_text = tk.Text(instr_frame, height=8, width=60, wrap=tk.WORD)
        instr_text.insert(tk.END,
                          "Supported commands:\n\n"
                          "- Next, Previous, First, Last, Exit\n"
                          "- Go to [number] or Jump to [number] (e.g., 'go to 5', 'jump to 10')\n"
                          "- The program uses PowerPoint's Ctrl+S shortcut to navigate to specific slides\n\n"
                          "Make sure PowerPoint is in presentation mode and has focus for the navigation to work.")
        instr_text.config(state=tk.DISABLED)
        instr_text.grid(row=0, column=0)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def update_languages(self):
        """Update the selected languages for recognition"""
        selected_langs = [lang for lang, var in self.lang_vars.items() if var.get()]
        if selected_langs:
            status = self.recognizer.set_target_words(selected_langs)
            self.status_var.set(status)
        else:
            self.status_var.set("Please select at least one language")

    def start_listening(self):
        """Start the voice recognition"""
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Start listening in a separate thread
        thread = threading.Thread(target=self.recognizer.listen_continuously,
                                  args=(self.update_status, self.log_command))
        thread.daemon = True
        thread.start()

    def stop_listening(self):
        """Stop the voice recognition"""
        self.recognizer.stop_listening()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_status(self, message):
        """Update the status label (thread-safe)"""

        def update():
            self.status_var.set(message)

        self.root.after(0, update)

    def log_command(self, text, command, confidence, success):
        """Add a command to the log (thread-safe)"""

        def add_log():
            timestamp = time.strftime("%H:%M:%S")
            confidence_str = f"{confidence:.2f}" if command else "N/A"
            command_str = command if command else "None"
            success_str = "Yes" if success else "No"

            self.log_tree.insert("", tk.END, values=(timestamp, text, command_str, confidence_str, success_str))
            # Auto-scroll to the bottom
            self.log_tree.yview_moveto(1)

        self.root.after(0, add_log)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceControlApp(root)
    root.mainloop()