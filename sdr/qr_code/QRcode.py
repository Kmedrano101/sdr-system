# -*- coding: utf-8 -*-
"""
Module for generate QR codes
Created on Mon Oct  3 18:46:07 2022

@author: Kmedrano101
Comments:
    - Check the possibility to use different QR codes on each bottle
    - content_qr: Needs to be related with a database
doc: Information capacity and versions of the QR code (https://www.qrcode.com/en/about/version.html)
    - Version 1     -> 21   modules - 128 bits / 16 Caracters
    - Version 2     -> 25   modules
    ...
    - Version 40    -> 177  modules
    
    12 _ 2bytes
    B 1byte
    
    
"""

# Import Modules
import qrcode
from PIL import Image
from utils.constants import QR_PATH, QR_NAME

# Define Class

class MyQRCode(qrcode.QRCode):
    """Class to generate QR code"""
    def __init__(self,version=1, box_size=2, border=1):
        self.__version = version
        self.__box_size = box_size
        self.__border = border
        self.__content = None
        self.error_correction = qrcode.constants.ERROR_CORRECT_H,
        super().__init__(version=version, box_size=box_size, border=border)

    # Define Properties
    @property
    def version(self):
        """The version property."""
        return self.__version
    @version.setter
    def version(self, value):
        self.__version = value

    @property
    def box_size(self):
        """The box_size property."""
        return self.__box_size
    @box_size.setter
    def box_size(self, value):
        self.__box_size = value

    @property
    def border(self):
        """The border property."""
        return self.__border
    @border.setter
    def border(self, value):
        self.__border = value

    @property
    def content(self):
        """The content property."""
        return self.__content
    @content.setter
    def content(self, value):
        self.__content = value

    # Define Functions
    def create_QR(self, fit=False) -> Image:
        """ Function to create a QR image"""
        self.add_data(self.content)
        self.make(fit)
        return self.make_image()
