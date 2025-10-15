for i in range(10):
    x, y = stream.next_instance()
    
    # ===== 单次调用完成所有事情 =====
    prediction = model.update(x, y)
    error = (prediction != y)
    errors.append(error)