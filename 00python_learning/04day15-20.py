# 简单选择排序
def select_sort(origin_item, comp=lambda x,y: x < y):
    items = origin_item[:]
    for i in range(len(items)-1):
        min_index = i
        for j in range(i+1, len(items)):
            if comp(items[j], items[min_index]):
                min_index = j
        items[i], items[min_index] = items[min_index], items[i]
    return items


# 冒泡排序（搅拌排序）
def bubble_sort(origin_items, comp=lambda x, y: x > y):
    items  = origin_items
    for i in range(len(items) - 1):
        swapped = False
        for j in range(i, len(items) - 1 - i):
            if comp(items[j], items[j+1]):
                items[j], items[j + 1] = items[j + 1], items[j]
                swapped = True
        if swapped:
            swapped = False
            for j in range(len(items) - 2 -i, i, -1):
                if comp(items[j - 1], items[j]):
                    items[j], items[j - 1] = items[j - 1], items[j]
                    swapped = True
        if not swapped:
            break
    return items


def main():
    origin_items = [12, 10, 2, 6, 8, 0, 4, 14]
    print(select_sort(origin_items))
    print(bubble_sort(origin_items))



if __name__ == '__main__':
    main()