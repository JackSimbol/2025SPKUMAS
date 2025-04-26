import numpy as np
import random


class Utils():
    
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.jokers = ['jo', 'jO']
    
    def Num2Poker(self, num: int):
        num_in_deck = num % 54
        if num_in_deck == 52:
            return "jo"
        if num_in_deck == 53:
            return "jO"
        # Normal cards:
        pokernumber = self.cardscale[num_in_deck // 4]
        pokersuit = self.suitset[num_in_deck % 4]
        return pokersuit + pokernumber
    
    def Poker2Num(self, poker: str, deck):
        num_in_deck = -1
        if poker[1] == "o":
            num_in_deck = 52
        elif poker[1] == "O":
            num_in_deck = 53
        else:
            num_in_deck = self.cardscale.index(poker[1])*4 + self.suitset.index(poker[0])
        if num_in_deck == -1:
            return -1
        if num_in_deck in deck:
            return num_in_deck
        return num_in_deck + 54
    
class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
    
    def __str__(self):
        return self.ErrorInfo  
    