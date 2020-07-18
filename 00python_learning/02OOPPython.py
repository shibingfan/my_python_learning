# class Cat:
#     count = 0
#
#     def __init__(self, name):
#         self.name = name
#         Cat.count += 1  #类属性
#
#     def __str__(self):
#         return "我是小猫：%s" % self.name
#
#     def eat(self):
#         print("%s 爱吃鱼" % self.name)
#
#     try:  # #异常处理
#         print(eat())
#     except ValueError:
#         print("请输入正确的整数")
#     except Exception as result:
#         print("未知错误 %s" % result)
#
#     @classmethod  # #类方法
#     def cat_count(cls):
#         print("Cat count = %d" % cls.count)
#
#     @staticmethod  # #静态方法  对象未创建前就可调用
#     def show_help():
#         print("帮助信息：让猫走进房间")
#
#
# tom = Cat("Tom")
# print(tom)
# tom.eat()
# print(Cat.count)  # 类属性调用
# tom.cat_count()  # 类方法调用
# Cat.cat_count()
# Cat.show_help()  #静态方法调用


class Person(object):

    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = ('_name', '_age', '_gender')

    def __init__(self, name, age):
        self._name = name
        self._age = age

    # 访问器-getter方法
    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self._age

    # 修改器-setter方法
    @age.setter
    def age(self, age):
        self._age = age

    def play(self):
        if self._age <= 16:
            print('%s正在玩飞行棋.' % self._name)
        else:
            print('%s正在玩斗地主.' % self._name)


def main():
    person = Person('王大锤', 12)
    print(person.age)
    person.play()
    person.age = 22
    print(person.age)
    person.play()


# 单例
class MusicPlayer(object):
    # 记录第一个被创建对象的引用
    instance = None
    # 记录是否执行过初始化动作
    init_flag = False

    def __new__(cls, *args, **kwargs):
        # 1. 判断类属性是否是空对象
        if cls.instance is None:
            # 2. 调用父类的方法，为第一个对象分配空间
            cls.instance = super().__new__(cls)
        # 3. 返回类属性保存的对象引用
        return cls.instance

    def __init__(self):
        if not MusicPlayer.init_flag:
            print("初始化音乐播放器")
            MusicPlayer.init_flag = True
