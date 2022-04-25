# train_step
total_iter += 1

# def training_step(data_iter, network, loss_fn, optimizer, args):
#     input, target = data_iter.next()
#     if args.cuda:
#         input = input.to(args.cuda_device)
#         target = target.to(args.cuda_device)

#     pred = network(input)
#     loss = loss_fn(pred, target)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

class Engine():
    pass