
from __future__ import annotations
# 어떤 것을 처리하기 위한 부품을 만들어버리자.
# 여러 부품을 만들 때 틀(클래스)을 먼저 만들어두고.
# 부품이 필요하면 틀을 찍어서 사용할 대상(부품을)을 만든다.
# Car 틀ㅇ르 먼저 만든다.
# --> 내차가 필요하면 Car틀을 이용해서 차를 만들고
# --> 내차의 특징을 상세하게 넣어줌.
# 만들 대상(부품, object, 객체)
# --> 대상(object, 객채)로 만들어서 코딩하는 방식
# --> 객체 지향형 프로그래밍(object Oriented Programming , OOP)

# 차에 대한 일반적인 특징을 만든다.
class Car:
    # 특징(속성)
    name : str
    price : int
    color : str
    def __init__(self, name = "현대", color = "검정색", price=3000):
        self.name = name
        self.color = color
        self.price = price

    def __str__(self):
        return f'{self.name} {self.color} {self.price}'

    #특징(동작)
    def run(self):
        print(f"{self.name} 달리다")

    def speed(self):
        print(f"{self. name} 스피트를 올리다")

    @staticmethod ## annotations(표시)
    def start():
        print('시동을 걸다')

if __name__ == "__main__":
    my_car = Car()
    print(my_car)

    my_car.price = 200
    my_car.color = "read"
    print(my_car)
    my_car.speed()
    Car.start()

    your_car = Car()
    print(your_car)
    your_car.speed()
