
def accu(fx, y):
    pred = fx.max(1, keepdim=True)[1]
    correct=pred.eq(y.view_as(pred)).sum()
    acc = correct.float()/pred.shape[0]
    return acc


def train(iterator, lossfunc, optimizer, model, device):

    epoch_acc = 0
    epoch_loss = 0
    model.train()
    for (x, y) in iterator:
    # for i, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        fx = model(x)
        loss = lossfunc(fx[0], y)
        loss.backward()
        optimizer.step()
        acc = accu(fx[0], y)

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(iterator, lossfunc, model, device):

    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    # for (x, y) in iterator:
    for i, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        fx = model(x)
        loss = lossfunc(fx[0], y)
        acc = accu(fx[0], y)

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)