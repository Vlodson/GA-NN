""" Nesto nije u redu sa poklapanjem dimenzija kod dot producta """


""" Importi """

from mlxtend.data import loadlocal_mnist #uvek sa jebenim mnistom sranja
import numpy as np



""" Database """

size = 10 #ovo menjaj za kolicinu slika
img, labels = loadlocal_mnist(images_path = "C:/Users/Vlada_Ana/Desktop/Vlada/Gen alg + NN/train-images.idx3-ubyte", labels_path = "C:/Users/Vlada_Ana/Desktop/Vlada/Gen alg + NN/train-labels.idx1-ubyte") #sad se ovako ucitava

img, labels = img[:size], labels[:size] # samo smanjujem br slika radi dobrobiti mene i racunara


""" Init param """

# neuroni

input_neurons = img.shape[0] # kolko slika
output_neurons = 10 # tolko cifara ima da se klasifikuje

hidden_layers = 3 # tako mi doslo (svakako ce kasnije menjati GA)
hidden_neurons = [5,10,5] # tako mi doslo (svakako ce kasnije menjati GA) OVO JE ZA TEST. BLOK ISPOD JE ZA PRAVU SITUACIJU

""" hidden_neurons = []
for i in range(hidden_layers):
    hidden_neurons.append(broj koji odabere GA)"""

# stvari za ucenje

learn_rate = 0.001 # tako mi doslo (svakako ce kasnije menjati GA)
learn_iter = 1000 # tako mi doslo (svakako ce kasnije menjati GA)

# weights i biases

def random_weights(layer1, layer2): # funkc koja samo pravi weightove za dva sloja
    weights = np.random.uniform(size = (layer1, layer2))
    return(weights)

w_ih = random_weights(input_neurons, hidden_neurons[0]) # input -> hidden

w_hh = []
for i in range(hidden_layers - 1):
    w_hh.append(random_weights(hidden_neurons[i], hidden_neurons[i+1])) # hidden -> hidden

w_ho = random_weights(hidden_neurons[-1], output_neurons) # hidden -> output

# print(w_ih)
# print(w_hh)
# print(w_ho)

def random_biases(layer):
    bias = np.random.uniform(size = (1, layer))
    return bias

b_h = []
for i in range(hidden_layers):
    b_h.append(random_biases(hidden_neurons[i]))

b_o = random_biases(output_neurons)


# aktivacione fukncije i njihovi izvodi (ovo ce odredjivati GA (negde cu ih staviti random da bira))

def tanhf(x):
    return np.tanh(x)

def d_tanhf(x):
    return 1 - pow(x,2)


# transfer

def transfer(layer1_values, layer2_weights, layer2_bias):
    trans = np.dot(layer1_values, layer2_weights) + layer2_bias # ovde mozda ne treba .T na weights
    return trans


""" Ucenje """

while i < learn_iter: # ne zaboravi i+=1


    """ Feedforward """

    # input -> hidden veza transfer i activation
    ih_transfer = transfer(img, w_ih, b_h[0])
    ih_activate = tanhf(ih_transfer)

    # hidden -> hidden veza transfer i activation
    hh_activate = []

    hh_transfer = transfer(ih_activate, w_hh[1], b_h[1]) # za posle input hidden veze mora ovako posto sam stavio gore ime ih_activate
    hh_activate.append(tanhf(hh_transfer))

    for i in range(hidden_layers-2): # ovde zajeb vrv (mislim da sam sredio)
        hh_transfer = transfer(hh_activate, w_hh[i+2], b_h[i+2]) # za ostale hh veze je ovako
        hh_activate.append(tanhf(hh_transfer))

    # hidden -> output veza transfer i activation
    ho_transfer = transfer(hh_activate, w_ho, b_o)
    ho_activate = tanhf(ho_transfer)

    # svi transferi bi trebalo da rade


    """ Backpropagation """


    # loss funkcija je L(x) = (label - output)^2 => je L'(x) = 2(label - output)
    d_loss_f = 2*(label - ho_activate)

    # izvodi
    d_ho_activate = d_tanhf(ho_activate) # hidden -> output veza

    d_hh_activate = [] # za sve hh veze
    for i in range(1, hidden_layers):
        d_hh_activate.append(d_tanhf(hh_activate[-(i+1)])) # ovde mi je izvod za h2 -> h1 vezu i za h1 -> h0 vezu tim redom

    d_ih_activate = d_tanhf(ih_activate) # za ih vezu (ne znam da li treba)

    # delta greska

    delta_ho = d_loss_f * d_ho_activate # ho veza

    delta_hh = []
    delta_hh.append(np.dot(delta_ho, w_ho.T) * d_hh_activate[0]) # ovde je d_hh_activate[0] zato sto d_hh_activate ide od nazad ka napred
                                                                 # i delta_hh ce opet ici od nazad

    for i in range(1, hidden_layers): # za ostale hh veze
        delta_hh.append(np.dot(delta_hh[i-1], w_hh[-i].T) * d_hh_activate[i])

    delta_ih = np.dot(delta_hh[-1], w_ih.T) * d_ih_activate


""" Test """


"""    Tacnost      """
