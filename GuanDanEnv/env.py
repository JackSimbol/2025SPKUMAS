import numpy as np
import random
from utils import Utils, Error
import warnings
from collections import Counter
    
class GuanDanEnv():
    '''
    Usage:
    Step1 Call GuanDanEnv() to create an instance
    Step2 Call GuanDanEnv.reset(config) to reset a match  
    '''
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Normaltypes = ("single", "pair", "three", "straight", "set", "three_straight", "triple_pairs")
        self.scaletypes = ("straight", "three_straight", "triple_pairs")
        self.Utils = Utils()
        self.level = None
        self.seed = None
        self.done = False
        self.game_state_info = "Init"
        self.cleared = [] # list of cleared players (have played all their decks)
        self.agent_names = ['player_%d' % i for i in range(4)]
        self.errset = {
            0: "Initialization Fault",
            1: "PlayerAction Fault",
            2: "Game Fault"
        }
        
    def reset(self, config={}):
        '''
        Call this function to start different matches
        @ config: contains match initalization info

        '''
        if 'seed' in config:
            self.seed = config['seed']
            random.seed(self.seed)
        if 'level' in config and config['level'] in self.cardscale:
            self.level = config['level']
        else:
            self.level = '2'
            warnings.warn("ResetConfigWarning: Level configuration fault or no level designated.")
            
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self._set_level()
        self.total_deck = [i for i in range(108)]
        self.card_todeal = [i for i in range(108)]
        random.shuffle(self.card_todeal)
        self.player_decks = [self.card_todeal[dpos*27 : (dpos+1) * 27] for dpos in range(4)]
        self.done = False
        self.history = []  
        self.round = 0
        self.played_cards = [[] for _ in range(4)]
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.pass_on = -1
        self.lastMove = {
            'player': -1, # the first round
            'action': [], 
            'claim': []
            }
        self.cleared = []
        self.game_state_info = "Running"
        return self._get_obs(0) # Each match starts from the player on 0 position
        
    def step(self, response):
        self.round += 1
        self.reward = None
        curr_player = response['player']
        action = response['action']
        claim = response['claim']
        if not self._is_legal_claim(action, claim): # not a legal claim
            self.game_state_info = f"Player {curr_player}: ILLEGAL CLAIM"
            return self._end_game(curr_player)
        for poker_no in action: 
            if poker_no in self.player_decks[curr_player]:
                self.player_decks[curr_player].remove(poker_no)
                self.played_cards[curr_player].append(poker_no)
            else:
                self.game_state_info = f"Player {curr_player}: NOT YOUR POKER"
                return self._end_game(curr_player)
        cur_pokertype, cur_points = self._check_poker_type(claim)
        if cur_pokertype == 'invalid':
            self.game_state_info = f"Player {curr_player}: INVALID TYPE"
            return self._end_game(curr_player)
        if len(self.lastMove['action']) == 0: # first-hand
            if cur_pokertype == 'pass':
                self.game_state_info = f"Player {curr_player}: ILLEGAL PASS AS FIRST-HAND"
                return self._end_game(curr_player)
            self.lastMove = response
            self.pass_on = -1
        else:
            if cur_pokertype != 'pass': # if currplayer passes, do nothing
                last_pokertype, last_points = self._check_poker_type(self.lastMove['claim'])
                bigger = self._check_bigger(last_pokertype, last_points, cur_pokertype, cur_points)
                if bigger == "error":
                    self.game_state_info = f"Player {curr_player}: POKERTYPE MISMATCH"
                    return self._end_game(curr_player)
                if not bigger:
                    self.game_state_info = f"Player {curr_player}: CANNOT BEAT LASTMOVE"
                    return self._end_game(curr_player)
                self.lastMove = response
                self.pass_on = -1
        
        self.history.append(response)
        if len(self.player_decks[curr_player]) == 0: # Finishing this round
            self.cleared.append(curr_player)
            if len(self.cleared) == 3: # match sealed
                self.done = True
                self.game_state_info = "Finished"
            elif len(self.cleared) == 2 and (self.cleared[1] - self.cleared[0]) % 2 == 0:
                self.done = True
                self.game_state_info = "Finished"
            self.pass_on = curr_player
            
        self._set_reward()
        
        if not self.done:
            next_player = (curr_player + 1) % 4
            if next_player == self.pass_on: # Successfully pass to teammate
                next_player = (self.pass_on + 2) % 4
                self.lastMove = {
                    "player": -1,
                    "action": [],
                    "claim": []
                }
            while next_player in self.cleared:
                next_player = (next_player + 1) % 4
                if next_player == self.pass_on: # Successfully pass to teammate
                    next_player = (self.pass_on + 2) % 4
                    self.lastMove = {
                        "player": -1,
                        "action": [],
                        "claim": []
                    }
            if next_player == self.lastMove['player']:
                self.lastMove = {
                    "player": -1,
                    "action": [],
                    "claim": []
                }
            return self._get_obs(next_player)
        return self._get_obs(-1) # Done. Send a signal to all players     
        
    
    def _set_reward(self):
        '''
        setting rewards
        if terminating: winner team gets reward 1~3
        else: rewards 0
        '''
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        if self.done:
            if len(self.cleared) == 2: # Must be a double-dweller
                self.reward[self.cleared[0]] = 3
                self.reward[self.cleared[1]] = 3
            elif (self.cleared[2] - self.cleared[0]) % 2 == 0:
                self.reward[self.cleared[0]] = 2
                self.reward[self.cleared[2]] = 2
            else:
                self.reward[self.cleared[0]] = 1
                self.reward[(self.cleared[0] + 2) % 4] = 1
    
    def _raise_error(self, errno, detail):
        raise Error(self.errset[errno]+": "+detail)
    
    def _end_game(self, fault_player):
        '''
        ending game on player's action exceptions
        '''
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.done = True
        self.reward[fault_player] = -3
        return self._get_obs(-1)
    
    def _get_obs(self, player):
        '''
        getting observation for player
        player: player_id (-1: all players)
        '''
        obs_set = {}
        for i in range(4):
            obs_set[i] = {
                "id": i,
                "level": self.level,
                "status": self.game_state_info,
                "deck": self.player_decks[i],
                "last_move": self.lastMove,
                "history": self.history,
                "reward": self.reward[i]
            }
        if player == -1:
            return obs_set
        else:
            return { player: obs_set[player] }
                
    def _set_level(self):     
        self.point_order.remove(self.level)
        self.point_order.append(self.level)
        self.point_order.extend(["o", "O"])
        
    def _is_legal_claim(self, action: list, claim: list):
        covering = "h" + self.level
        if len(action) != len(claim):
            return False
        action_pok = [self.Utils.Num2Poker(p) for p in action]
        claim_pok = [self.Utils.Num2Poker(p) for p in claim]
        for pok in action_pok:
            if pok != covering:
                if pok in claim_pok:
                    claim_pok.remove(pok)
                else:
                    return False
        for pok in claim_pok:
            if pok[1] == 'o' or pok[1] == 'O':
                return False
        return True
    
    def _check_poker_type(self, poker: list):
        if poker == []:
            return "pass", ()
        # covering = "h" + level
        poker = [self.Utils.Num2Poker(p) for p in poker]
        if len(poker) == 1:
            return "single", (poker[0][1])
        if len(poker) == 2:
            if poker[0][1] == poker[1][1]:
                return "pair", (poker[0][1])
            return "invalid", ()
        # 大于等于三张
        points = [p[1] for p in poker]
        cnt = Counter(points)
        vals = list(cnt.values())
        if len(poker) == 3:
            if "o" in points: 
                return "invalid", ()
            if vals.count(3) == 1:
                return "three", (points[0])
            return "invalid", ()
        if len(poker) == 4: # should be a bomb
            if "o" in points or "O" in points: # should be a rocket
                if cnt["o"] == 2 and cnt["O"] == 2:
                    return "rocket", ("jo")
                return "invalid", ()
            if vals.count(4) == 1:
                return "bomb", (4, points[0])
            return "invalid", ()
        if len(poker) == 5: # could be straight, straight flush, three&two or bomb
            if vals.count(5) == 1:
                return "bomb", (5, points[0])
            if vals.count(3) == 1 and vals.count(2) == 1: # set: 三带二 
                three = ''
                two = ''
                for k in list(cnt.keys()):
                    if cnt[k] == 3:
                        three = k
                    elif cnt[k] == 2:
                        two = k
                return "set", (three, two)
            if vals.count(1) == 5: # should be straight
                points.sort(key=lambda x: self.cardscale.index(x))
                suits = [p[0] for p in poker]
                suit_cnt = Counter(suits)
                suit_vals = list(suit_cnt.values())
                flush = False
                if suit_vals.count(5) == 1:
                    flush = True
                first = points[0]
                if first == 'A':
                    if points == ['A', '0', 'J', 'Q', 'K']:
                        if flush:
                            return "straight_flush", ('0')
                        return "straight", ('0')
                sup_straight = [self.cardscale[self.cardscale.index(first)+i] for i in range(5)]
                if points == sup_straight:
                    if flush:
                        return "straight_flush", (first)
                    return "straight", (first)
            return "invalid", ()
        if len(poker) == 6: # could be triple_pairs, three_straight, bomb
            if vals.count(6) == 1:
                return "bomb", (6, points[0])
            if vals.count(3) == 2:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', '2']:
                        return "three_straight", ('A')
                    if ks == ['A', 'K']:
                        return "three_straight", ('K')
                    return "invalid", ()
                if self.cardscale.index(ks[1]) - self.cardscale.index(ks[0]) == 1:
                    return "three_straight", (ks[0])
            if vals.count(2) == 3:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', 'Q', 'K']:
                        return "triple_pairs", ('Q')
                    if ks == ['A', '2', '3']:
                        return "triple_pairs", ('A')
                    return "invalid", ()
                pairs = [self.cardscale[self.cardscale.index(ks[0])+i] for i in range(3)]
                if ks == pairs:
                    return "triple_pairs", (ks[0])
            return "invalid", ()
        if len(poker) > 6 and len(poker) <= 10:
            if vals.count(len(poker)) == 1:
                bomb = points[0]
                return "bomb", (len(poker), bomb)
        return "invalid", ()
    
    def _check_bigger(self, type1, point1, type2, point2):
        '''
        Check if poker2(type2, point2) is bigger than poker1(type1, point1)
        Assumption: type1 and type2 are VALID cardtypes. Must check types before calling this function 
        '''
        if type2 == "rocket":
            return True
        if type1 == "rocket":
            return False
        if type1 in self.Normaltypes:
            if type2 not in self.Normaltypes:
                return True
            if type1 == type2:
                if type1 in self.scaletypes and self.cardscale.index(point2[0]) > self.cardscale.index(point1[0]):
                    return True
                if type1 not in self.scaletypes and self.point_order.index(point2[0]) > self.point_order.index(point1[0]):
                    return True
                return False
            return "error"
        if type2 in self.Normaltypes:
            return "error"
        if type1 == "bomb":
            if type2 == "bomb":
                if point2[0] == point1[0] and self.point_order.index(point2[1]) > self.point_order.index(point1[1]):
                    return True
                if point2[0] > point1[0]:
                    return True
            if type2 == "straight_flush":
                if point1[0] < 6:
                    return True
        if type1 == "straight_flush":
            if type2 == "bomb":
                if point2[0] >= 6:
                    return True
                return False
            if type2 == "straight_flush":
                if self.cardscale.index(point2[0]) > self.cardscale.index(point1[0]):
                    return True
        return False
    