from __future__ import annotations

class Brick:
    size: int
    color: str
    name: str
    def __init__(self, size = 10, color='red', name = "벽돌1"):

        self.size = size
        self.color = color
        self.name = name

    def __str__(self):
        return f'{self.name} {self.size} {self.color}'

    def build(self):
        print(f"{self.name}을 쌓다")

    @staticmethod
    def tap():
        print("벽돌을 두드리다")


if __name__ == '__main__':
    bk1 = Brick()
    bk1.size = 100
    bk1.color = 'red'
    print(bk1)
    bk1.build()

    bk2 = Brick()
    bk2.size = 100
    bk2.color = 'blue'
    bk2.name = '벽돌2'
    print(bk2)
    bk2.build()

    Brick.tap()