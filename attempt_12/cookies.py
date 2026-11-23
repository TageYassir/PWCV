# ======================
# USER PREFERENCES (COOKIES) SYSTEM
# ======================
import json
from pathlib import Path
import os


class UserPreferences:
    def __init__(self):
        # Get the directory where the script is located
        script_dir = Path(__file__).parent
        self.preferences_file = script_dir / "user_preferences.json"
        self.default_preferences = {
            "theme": "Default",
            "mode": "custom",
            "slide_delay": 1.0,
            "base_zoom_factor": 1.5,
            "dynamic_zoom": True,
            "pointer_speed": 2.5,
            "validate_posture": True,
            "enable_toggle_mode_nearby": True,
            "enable_toggle_mode_conference": False,
            "camera_mode": "normal",
            "enable_pose_estimation": False,
            "enable_furniture_detection": False,
            "teacher_mode_active": True,
            "enable_voice_control": False,
            "voice_language": "en"
        }
        self.current_preferences = self.default_preferences.copy()

    def load_preferences(self):
        """Load user preferences from file"""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    loaded_prefs = json.load(f)
                    # Update current preferences with loaded ones
                    self.current_preferences.update(loaded_prefs)
                    return True
        except Exception as e:
            print(f"Error loading preferences: {e}")
        return False

    def save_preferences(self):
        """Save current preferences to file"""
        try:
            # Create directory if it doesn't exist
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.preferences_file, 'w') as f:
                json.dump(self.current_preferences, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
        return False

    def get(self, key, default=None):
        """Get a preference value"""
        return self.current_preferences.get(key, default)

    def set(self, key, value):
        """Set a preference value"""
        self.current_preferences[key] = value
        self.save_preferences()

    def apply_preferences(self, app):
        """Apply saved preferences to the application"""
        # Set flag to indicate we're applying preferences
        app._applying_preferences = True

        # Theme
        if hasattr(app, 'theme_var'):
            app.theme_var.set(self.get('theme', 'Default'))
            # Don't call change_theme() here to avoid recursion

        # Mode
        if hasattr(app, 'mode_var'):
            app.mode_var.set(self.get('mode', 'custom'))

        # Slide delay
        app.slide_delay = self.get('slide_delay', 1.0)
        if hasattr(app, 'delay_slider'):
            app.delay_slider.set(app.slide_delay)

        # Zoom settings
        app.base_zoom_factor = self.get('base_zoom_factor', 1.5)
        app.dynamic_zoom = self.get('dynamic_zoom', True)
        if hasattr(app, 'zoom_slider'):
            app.zoom_slider.set(app.base_zoom_factor)
        if hasattr(app, 'dynamic_zoom_var'):
            app.dynamic_zoom_var.set(app.dynamic_zoom)

        # Pointer speed
        app.pointer_speed_multiplier = self.get('pointer_speed', 2.5)
        if hasattr(app, 'speed_slider'):
            app.speed_slider.set(app.pointer_speed_multiplier)

        # Posture validation
        app.validate_posture = self.get('validate_posture', True)
        if hasattr(app, 'posture_validation_var'):
            app.posture_validation_var.set(app.validate_posture)

        # Toggle modes
        app.enable_toggle_mode_nearby = self.get('enable_toggle_mode_nearby', True)
        app.enable_toggle_mode_conference = self.get('enable_toggle_mode_conference', False)
        if hasattr(app, 'toggle_mode_nearby_var'):
            app.toggle_mode_nearby_var.set(app.enable_toggle_mode_nearby)
        if hasattr(app, 'toggle_mode_conference_var'):
            app.toggle_mode_conference_var.set(app.enable_toggle_mode_conference)

        # Camera mode
        if hasattr(app, 'camera_mode_var'):
            app.camera_mode_var.set(self.get('camera_mode', 'normal'))
            if app.camera_mode_var.get() == "high":
                app.set_camera_high_mode()
            else:
                app.set_camera_normal_mode()

        # Pose estimation
        app.enable_pose_estimation = self.get('enable_pose_estimation', False)
        if hasattr(app, 'pose_estimation_var'):
            app.pose_estimation_var.set(app.enable_pose_estimation)

        # Furniture detection
        app.enable_furniture_detection = self.get('enable_furniture_detection', False)
        if hasattr(app, 'furniture_detection_var'):
            app.furniture_detection_var.set(app.enable_furniture_detection)

        # Teacher mode
        app.teacher_mode_active = self.get('teacher_mode_active', True)
        if hasattr(app, 'teacher_mode_var'):
            app.teacher_mode_var.set(app.teacher_mode_active)

        # Voice control - only apply if the attributes exist
        if hasattr(app, 'enable_voice_control'):
            app.enable_voice_control = self.get('enable_voice_control', False)

        if hasattr(app, 'voice_control_var'):
            app.voice_control_var.set(app.enable_voice_control)

        # Voice language - only apply if the attribute exists
        if hasattr(app, 'voice_lang_var'):
            app.voice_lang_var.set(self.get('voice_language', 'en'))

        # Voice language
        app.voice_lang_var.set(self.get('voice_language', 'en'))

        # Clear the flag
        app._applying_preferences = False

        # Now apply the theme after all other preferences
        if hasattr(app, 'theme_var'):
            app.change_theme()