# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# ===============================================================================================================

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(f'out: \n{out}')
print(f'hidden: \n{hidden}')

# ===============================================================================================================

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# ===============================================================================================================

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
# ===============================================================================================================

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(f'tag_scores before training: \n{tag_scores}')

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(f'tag_scores after training: \n{tag_scores}')

# ===============================================================================================================


class LSTMown(L.LightningModule):
    def __init__(self): # Create and init weight and bias tensors
        super().__init__()
        # Use normal distribution to randomly generate the first weight
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # lr = long remember
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # Parameter for the first bias, set to 0
        self.blr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        # pr = percent remember
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # Parameter for the first bias, set to 0
        self.bpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        # p = potential
        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # Parameter for the first bias, set to 0
        self.bp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        #
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # Parameter for the first bias, set to 0
        self.bo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory): # LSTM math
        long_remember_percent = torch.sigmoid((short_memory*self.wlr1)+(input_value*self.wlr2)+self.blr1)

        potential_remember_percent = torch.sigmoid((short_memory+self.wpr1)+(input_value*self.wpr2)+self.bpr1)

        potential_memory = torch.tanh((short_memory+self.wp1)+(input_value*self.wp2)+self.bp1)

        update_long_memory = ((long_memory*long_remember_percent)+(potential_remember_percent*potential_memory))

        output_percent = torch.sigmoid((short_memory*self.wo1)+(input_value*self.wo2)+self.bo1)

        update_short_memory = torch.tanh(update_long_memory)*output_percent

        return ([update_long_memory, update_short_memory])


    def forward(self, input): # forward pass to unrolled LSTM
        long_memory = 0
        short_memory = 0

        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self): # Config Adam
        return Adam(self.parameters())

    def training_step(self, batch, batch_index): # Calculate loss and log 
        input_i, label_i = batch

        output_i = self.forward(input_i[0])

        loss = (output_i - label_i)**2

        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
        return loss

model = LSTMown()

print(f'Company A: Oberved = 0, Predicted = {model(torch.tensor([0.,0.5, 0.25, 1.])).detach()}')
print(f'Company A: Oberved = 1, Predicted = {model(torch.tensor([1.,0.5, 0.25, 1.])).detach()}')

inputs = torch.tensor([[0.,0.5, 0.25, 1.], [1.,0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)
'''
DataLoader are usefull because they make it easy to ...
1. ... access data
2. ... shuffle data each epoch
3. ... use a relatively small fraction of the data for debugging
'''

trainer = L.Trainer(max_epochs=2000)

trainer.fit(model, train_dataloaders=dataloader)

print(f'Company A: Oberved = 0, Predicted = {model(torch.tensor([0.,0.5, 0.25, 1.])).detach()}')
print(f'Company A: Oberved = 1, Predicted = {model(torch.tensor([1.,0.5, 0.25, 1.])).detach()}')

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path

trainer = L.Trainer(max_epochs=3000)
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

print(f'Company A: Oberved = 0, Predicted = {model(torch.tensor([0.,0.5, 0.25, 1.])).detach()}')
print(f'Company A: Oberved = 1, Predicted = {model(torch.tensor([1.,0.5, 0.25, 1.])).detach()}')

# --------------------

class LightningLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1) # hidden size refers to the number of output values we want

    def forward(self, input):
        intput_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(intput_trans) # contains the STM values from each LSTM unit that we want

        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_index): # Calculate loss and log 
        input_i, label_i = batch

        output_i = self.forward(input_i[0])

        loss = (output_i - label_i)**2

        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
        return loss
    
model = LightningLSTM()
print(f'Company A: Oberved = 0, Predicted = {model(torch.tensor([0.,0.5, 0.25, 1.])).detach()}')

trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)

trainer.fit(model, train_dataloaders=dataloader)
print(f'Company A: Oberved = 0, Predicted = {model(torch.tensor([0.,0.5, 0.25, 1.])).detach()}')
