from src.UI.InterfaceUsuarioFactory import InterfaceUsuarioFactory
from src.perceptron.PerceptronRU import PerceptronRU
from src.services.AmostrasServico import AmostrasServico

# Exemplo de uso
if __name__ == "__main__":

    #Inicializa a UI de interação com o User. Obs : Só está implementado a console UI
    interface_usuario_comunicacao = InterfaceUsuarioFactory().criar_interface_usuario()
    
    # Obter RU do Aluno através da UI
    ru_aluno_referencia = interface_usuario_comunicacao.ler("\nDigite o seu RU para usar o mesmo como referência do percetron")

     # Criar e treinar o perceptron
    perceptron = PerceptronRU(interface_usuario=interface_usuario_comunicacao,
                              taxa_aprendizado=0.001,
                              num_entradas= len(ru_aluno_referencia),
                              ru_aluno_referencia= ru_aluno_referencia)

    # Gerar conjuntos de dados
    conjunto_treinamento = AmostrasServico().gerar_conjunto_treinamento(ru_aluno_referencia)
    conjunto_teste = AmostrasServico().gerar_conjunto_teste()
    
    # Treinar
    perceptron.treinar(conjunto_treinamento, max_epocas=1000)
    
    # Testar
    perceptron.testar(conjunto_teste)
    
    interface_usuario_comunicacao.escrever("=== ANALISE FINAL ===")
    interface_usuario_comunicacao.escrever(f"RU do Aluno: {perceptron.ru_referencia}")
    interface_usuario_comunicacao.escrever(f"Pesos finais: {[round(w, 3) for w in perceptron.pesos]}")
    interface_usuario_comunicacao.escrever(f"Bias final: {round(perceptron.bias, 3)}")
    
    # Teste manual com o próprio RU
    interface_usuario_comunicacao.escrever(f"\nTeste com o próprio RU {perceptron.ru_referencia}:")
    resultado = perceptron.predizer(perceptron.ru_referencia)
    interface_usuario_comunicacao.escrever(f"Resultado: {resultado:+d} (esperado: 1)")

    # Teste manual com diversos outros RU's para validar a assetividade do percetron
    perceptron.testar_assertividade()
