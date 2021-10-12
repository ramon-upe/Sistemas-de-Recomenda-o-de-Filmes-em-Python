#Importação da livraria NumPy
import numpy as np
#Indicando o repositorio e importando o arquivo
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#buscar dados do modelo
data = fetch_movielens(min_rating = 4.0)

#imprimir treinamento e dados de teste
print('Tamanho da base de Treinamento')
print(repr(data['train']))
print('Tamanho da base de Teste')
print(repr(data['test']))
print('---------------------------------------------------------------------')

#criando o  modelo
model = LightFM(loss = 'warp')

#modelo de treinamento
model.fit(data['train'], epochs=30, num_threads=2)

#Intanciando função de recomendação
def sample_recommendation(model, data, user_ids):
    #número de usuários e filmes nos dados de treinamento
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
    	#filmes que foram avaliados como 'gostei'
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #filmes que nosso modelo prevê que eles vão gostar
        scores = model.predict(user_id, np.arange(n_items))
        #classificandu-os na ordem do mais gostado para o menos
        top_items = data['item_labels'][np.argsort(-scores)]
        #imprimir os resultados
        print("Usuario: %s" % user_id)
        print("     Filmes que o usuario %s" % user_id + " Gostou!")

        for x in known_positives[:5]:
            print("        %s" % x)

        print("     Titulos Recomendados de acordo com o seu perfil:")

        for x in top_items[:5]:
            print("        %s" % x)
            
sample_recommendation(model, data, [3, 25, 451])





