# # hello world
# print("Hello Python")  # #shibinfgan

# #变量使用
# price = float(input("输入价格:"))
# print("价格是：%.2f" %price)

# #判断循环
# import random
#
# for i in range(10):
#     print(random.randrange(1, 4, 1), end=' ')
#
# j = k = 0
# while j == 0:
#     if k == 5:  # 最多玩5局
#         print("game over")
#         break
#     else:
#         player = int(input("请出石头1，剪刀2，布3:"))
#         computer = int(random.randrange(1, 4, 1))
#         if ((player == 1 and computer == 2) or
#                 (player == 2 and computer == 3) or
#                 (player == 3 and computer == 1)):
#             print("you win,the computer is %d" % computer)
#             j = 1
#         elif player == computer:
#             print("try again,the computer also is %d" % computer)
#         else:
#             print("you lost,the computer is %d" % computer)
#         k += 1

# # 函数基础
# def print_table():
#     i = 1
#     while i <= 9:
#         j = 1
#         while j <= i:
#             print("%d*%d = %d" % (j, i, i*j ), end = '\t')
#             j += 1
#         print()
#         i += 1
# print_table()

# # 名片管理系统
# card_list = []
# def show_menu():
#     print("*" * 50)
#     print("欢迎使用【菜单管理系统】V1.0")
#     print("1. 新建名片")
#     print("2. 显示全部")
#     print("3. 查询名片")
#     print("0. 退出系统")
#     print("*" * 50)
#
# def new_card():
#     name = input("name:")
#     phone = input("phone:")
#     qq = input("QQ:")
#     email = input("email:")
#
#     card_dict = {"name":name, "phone":phone, "qq":qq, "email":email}
#     card_list.append(card_dict)
#     print("new card %s had been added" % card_dict["name"])
#
# def show_all():
#     print("-" * 50)
#     if len(card_list) == 0:
#         print("there is no card")
#     else:
#         for title in ["name", "phone", "QQ", "email"]:
#             print(title, end="\t\t")
#         print()
#         for card in card_list:
#             print("%s\t\t%s\t\t%s\t\t%s" % (card["name"], card["phone"], card["qq"], card["email"]))
#
# def search_card():
#     find_name = input("input the search name:")
#     for card_dict in card_list:
#         if card_dict["name"] == find_name:
#             for title in ["name", "phone", "QQ", "email"]:
#                 print(title, end="\t\t")
#             print()
#             print("%s\t\t%s\t\t%s\t\t%s" % (card_dict["name"], card_dict["phone"], card_dict["qq"], card_dict["email"]))
#             break
#         else:
#             print("there is no %s" % find_name)
#
# show_menu()
# while True:
#     action = input("choose option:")
#     print("your choice is: %s" % action)
#
#     if action == "1":
#         new_card()
#     elif action == "2":
#         show_all()
#     elif action == "3":
#         search_card()
#     elif action == "0":
#         print("welcome to use next time")
#         break
#     else:
#         print("something wrong, please choose again")


def sum_n(num):
    if num == 1:
        return 1
    return num + sum_n(num-1)
print(sum_n(5))

