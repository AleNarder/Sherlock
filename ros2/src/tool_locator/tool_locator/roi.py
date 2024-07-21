import cv2
import numpy as np

class ROIWindow ():
    
    def __init__(self, frame_w, frame_h, win_w = None, win_h = None) -> None:
        self.frame_w_  = frame_w 
        self.frame_h_  = frame_h
        self._padding_ = 0
        self._width    = win_w if win_w is not None else int(frame_w * 0.33)
        self._height   = win_h if win_h is not None else int(frame_h * 0.75)
        self._left     = int(frame_w / 2 - self._width  / 2)
        self._top      = int(frame_h / 2 - self._height / 2)
        self._color    = (255, 0, 0)

    def set_color (self, color):
        self._color = color

    def calibrate_width(self, width):
        self._width = width
    
    def calibrate_height(self, height):
        self._height = height

    def calibrate_offset_x(self, offset):
        self._left += offset
    
    def calibrate_offset_y(self, offset):
        self._top+= offset 

    def get_window (self, frame: np.array) -> np.array:
        return frame.copy()[self._top : self._top + self._height, self._left:self._left + self._width]

    def apply_window (self, frame: np.array, in_place = True) -> np.array:
        """
        """
        cloned = frame.copy() if not in_place else frame
        left   = self._left - self._padding_
        top    = self._top  - self._padding_
        right  = self._left + self._width  + self._padding_
        bottom = self._top  + self._height + self._padding_
        cv2.rectangle(cloned, (left + 1, top + 1), (right - 1, bottom - 1), self._color, 1)
        return cloned
    
    def apply_window_patch(self, frame: np.array, patch: np.array, in_place = True):
        """
        """
        cloned = frame.copy() if not in_place else frame
        cloned[self._top : self._top + self._height, self._left:self._left + self._width] = patch
        return cloned

    def apply_hr_calib_line (self, frame: np.array, offset: int, from_bottom = False, in_place = True):
        """
        """
        cloned = frame.copy() if not in_place else frame
        offset = offset if not from_bottom else self.frame_h_ - offset
        cv2.line(cloned, (0,  offset), (self.frame_w_, offset), self._color, 1)
        return cloned

    def apply_vr_calib_line (self, frame: np.array, offset: int, from_right = False, in_place = True):
        """
        """
        cloned = frame.copy() if not in_place else frame
        offset = offset if not from_right else self.frame_w_ - offset
        cv2.line(cloned, (offset, 0), (offset, self.frame_h_), self._color, 1)
        return cloned
    
    