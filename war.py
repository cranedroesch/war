import queue
import random

class War:
    def __init__(self, seed=None, verbose=False):
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)
        self.deck = [i for i in range(1, 14) for j in range(4)]
        self.shuffle()
        self.p1, self.p2 = self.deal()
        self.turns = 0

    def shuffle(self):
        random.shuffle(self.deck)
        assert len(self.deck) == 52
        assert len(set(self.deck)) == 13

    def deal(self):
        p1, p2 = queue.Queue(52), queue.Queue(52)
        for card in range(0, 26):
            p1.put(self.deck[card], False)
            p2.put(self.deck[card + 26], False)
        assert p1.qsize() == p2.qsize() == 26
        return p1, p2

    def add_to_pile(self, pile, pot):
        if self.verbose:
            print(f"Adding pot {pot} to pile {pile} of size {pile.qsize()}")
        random.shuffle(pot)
        for card in pot:
            pile.put(card, False)

    def end_game(self):
        if self.verbose:
            print("GAME OVER")
            print(self.status_string())
        exit()

    def fight(self):
        if self.verbose:
            print(f"Starting turn {self.turns}")
        if self.p1.empty() | self.p2.empty():
            self.end_game()
        try:
            c1 = self.p1.get(False)
        except queue.Empty:
            self.end_game()
        try:
            c2 = self.p2.get(False)
        except queue.Empty:
            self.p1.put(c1, False)  # p1 still has original card
            self.end_game()
        pot = [c1, c2]
        if c1 > c2:
            if self.verbose:
                print(f"{c1}, {c2}, p1 wins {pot}")
            self.add_to_pile(self.p1, pot)
        elif c2 > c1:
            if self.verbose:
                print(f"{c1}, {c2}, p2 wins {pot}")
            self.add_to_pile(self.p2, pot)
        elif c1 == c2:  # war
            if self.verbose:
                print(f"WAR!!! {c1}, {c2}, pot {pot}")
            self.war(pot)
        if self.verbose:
            print(f"Turn end status: {self.status_string()}")
        try:
            assert self.p1.qsize() + self.p2.qsize() == 52
        except AssertionError as error:
            print(f"Error: {error}")
            breakpoint()

    def war(self, pot):
        if self.verbose:
            print(f"WAR! Pot: {pot}")
        card1, card2 = 999, 999
        for card in range(0, 4):  # last time is the comparison card
            try:
                tempCard = self.p1.get(False)
                card1 = tempCard
                pot.append(tempCard)
                if self.verbose:
                    print(f"Added {tempCard} to pot; pot now {pot}")
            except queue.Empty:
                if card1 == 999:
                    self.add_to_pile(self.p2, pot)
                    self.end_game()
            try:
                tempCard = self.p2.get(False)
                card2 = tempCard
                pot.append(tempCard)
                if self.verbose:
                    print(f"Added {tempCard} to pot; pot now {pot}")
            except queue.Empty:
                if card2 == 999:
                    self.add_to_pile(self.p1, pot)
                    self.end_game()
        if card1 > card2:
            if self.verbose:
                print(f"{card1}, {card2}: p1 wins pot {pot}")
            self.add_to_pile(self.p1, pot)
            if self.verbose:
                print(self.status_string())
            return
        elif card2 > card1:
            if self.verbose:
                print(f"{card1}, {card2}: p2 winse pot {pot}")
            self.add_to_pile(self.p2, pot)
            if self.verbose:
                print(self.status_string())
            return
        else:
            return self.war(pot)

    def status_string(self):
        return f"turn {self.turns} p1: {self.p1.qsize()}, p2: {self.p2.qsize()}"

    def game(self):
        while (not self.p1.empty()) & (not self.p2.empty()):
            self.turns += 1
            try:
                self.fight()
            except BaseException as error:
                if self.p1.empty() | self.p2.empty():
                    continue
                print(f"Some other error: {error}")
                print(self.status_string())
                breakpoint()
            if (self.turns % 100 == 0) & self.verbose:
                print(f"Turns % 100 == 0: {self.status_string()}")
        if self.verbose:
            print(f"GAME OVER: {self.status_string()}")
        return self.turns


if __name__ == "__main__":
    turn_counts = []
    for n in range(10):
        game = War(verbose=False)
        turn_counts.append(game.game())
    print(f"Turn counts: {turn_counts}")