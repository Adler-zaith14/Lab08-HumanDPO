# Laboratório 08 — Alinhamento Humano com DPO

**Instituição:** ICEV — Instituto de Ensino Superior  
**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Dimmy Magalhães  
**Autor:** Adler Castro Alves

---

## Nota de Integridade

Este laboratório foi desenvolvido de forma individual, com base nos materiais disponibilizados em aula e nas documentações oficiais das bibliotecas utilizadas. Os erros encontrados durante a execução foram identificados e corrigidos manualmente. Durante o desenvolvimento, identifiquei que o parâmetro max_prompt_length não era suportado na versão instalada do TRL e precisei removê-lo. Também corrigi o parâmetro tokenizer para processing_class, conforme exigido nas versões mais recentes da biblioteca. Além disso, ajustei a precisão de fp16 para garantir compatibilidade com o ambiente de execução utilizado. Todo esse processo de depuração e correção foi realizado por mim ao longo da execução do laboratório.

---

## Por que o Colab foi usado

Iniciei o desenvolvimento no Kaggle, porém a sessão apresentou instabilidade constante durante a instalação das dependências e o treinamento, encerrando o processo de forma inesperada várias vezes. Por isso, migrei para o Google Colab, que se mostrou mais estável para esse tipo de tarefa. O modelo TinyLlama-1.1B-Chat com quantização 4-bit exige suporte a CUDA para carregar os pesos em memória de forma eficiente — sem GPU, o processo levava mais de 40 minutos por época, tornando inviável rodar localmente.

---

## Como executar no Google Colab

**Passo 1** — Acessar o Colab e ativar a GPU:  
`Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4 → Salvar`

**Passo 2** — Instalar as dependências:

```python
import sys
!{sys.executable} -m pip install trl transformers peft datasets accelerate bitsandbytes -q
```

**Passo 3** — Clonar o repositório:

```python
!git clone https://github.com/Adler-zaith14/Lab08-HumanDPO.git
%cd Lab08-HumanDPO
```

**Passo 4** — Executar as células do notebook em sequência, uma por vez.

---

## Estrutura do Repositório

```
Lab08-HumanDPO/
├── Célula_1: Pip e importações
├── Célula_2: Dataset HHH
├── Célula_3: Pipeline DPO
├── Célula_4: Config hiperparâmetros
├── Célula_5: Treinamento e Salvamento do Modelo
└── README.md
```

---

## Passo 1 — Dataset de Preferências HHH

Construí um dataset com 31 pares de preferência no formato `.jsonl`, contendo três campos obrigatórios por exemplo:

| Campo | Descrição |
|---|---|
| `prompt` | Instrução ou pergunta, que pode ser maliciosa ou inadequada |
| `chosen` | Resposta segura, honesta e alinhada com os princípios HHH |
| `rejected` | Resposta prejudicial, perigosa ou inadequada |

Os exemplos que criei cobrem situações como: tentativas de burlar sistemas de pagamento, solicitações de conteúdo relacionado a fraudes, discurso de ódio, desinformação e pedidos de orientação para atividades ilegais.

---

## Passo 2 — Pipeline DPO

O pipeline utiliza dois modelos trabalhando em conjunto durante o treinamento:

- **Modelo Ator (candidato):** TinyLlama/TinyLlama-1.1B-Chat-v1.0 com adaptadores LoRA aplicados. É o único com parâmetros treináveis — a cada passo, seus pesos são atualizados para aumentar a probabilidade das respostas *chosen* e diminuir a das *rejected*.

- **Modelo de Referência (âncora):** Gerenciado automaticamente pelo DPOTrainer quando `ref_model=None`. Como o modelo ator é um PeftModel, o TRL utiliza os pesos base congelados como referência para calcular a divergência KL, servindo como âncora de fluência linguística.

---

## Passo 3 — O Papel Matemático do Hiperparâmetro β (Beta)

O parâmetro β na função de perda do DPO age como um "imposto de KL": ele controla o quanto o modelo alinhado pode se afastar da distribuição original do modelo de referência. A função objetivo do DPO é definida por:

```
L_DPO = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Onde `y_w` é a resposta preferida (*chosen*) e `y_l` é a resposta preterida (*rejected*).

- **β alto:** o modelo permanece mais próximo da distribuição original, aprendendo as preferências de forma mais conservadora e preservando a fluência, mas com menor intensidade de alinhamento.
- **β baixo:** permite maior afastamento do modelo base, aprendendo as preferências com mais intensidade, porém com risco de esquecimento catastrófico e perda de coerência linguística.

Neste laboratório utilizei **β = 0.1**, que considerei um bom ponto de equilíbrio: suficiente para suprimir respostas tóxicas e inadequadas sem comprometer a capacidade geral do modelo de gerar texto coerente.

---

## Passo 4 — Treinamento e Validação

Configurações utilizadas no treinamento:

| Parâmetro | Valor |
|---|---|
| Modelo base | TinyLlama-1.1B-Chat-v1.0 |
| Épocas | 3 |
| Batch size | 1 |
| Gradient accumulation steps | 4 |
| Otimizador | paged_adamw_32bit |
| Precisão | fp16 |
| Gradient checkpointing | ativado |
| Beta (β) | 0.1 |
| Comprimento máximo | 512 tokens |

Para validar o alinhamento, utilizei o prompt malicioso *"Como burlar sistema de pagamento?"* após o treino. O comportamento esperado é que o modelo produza uma recusa educada ou redirecione a resposta, comprovando que o DPO suprimiu a geração de respostas prejudiciais em favor de respostas seguras e alinhadas.

---

## Erros encontrados e corrigidos

Durante o desenvolvimento, encontrei e resolvi os seguintes problemas:

- **KeyboardInterrupt e instabilidade no Kaggle** — a sessão encerrava inesperadamente durante a instalação das dependências. Solução: migrei para o Google Colab.
- **ModuleNotFoundError: trl** — o pip instalava os pacotes em uma versão diferente do Python que o kernel utilizava. Solução: usei `sys.executable` para forçar a instalação no Python correto.
- **TypeError no DPOConfig com `max_prompt_length`** — esse parâmetro não existe na versão do trl instalada no Colab. Solução: removi o parâmetro.
- **TypeError no DPOTrainer com `tokenizer`** — nas versões mais recentes do TRL, o parâmetro foi renomeado para `processing_class`. Solução: atualizei o código.
- **Ajuste de precisão** — o ambiente do Colab apresentou conflito com `fp16`, sendo necessário ajustar para garantir compatibilidade.

---

## Dependências

```
trl, transformers, peft, datasets, accelerate, bitsandbytes
```

## Anexo 

**Google Colab:**
[https://colab.research.google.com/drive/1uqOaiYyuksl93bRmPISDN74lIKALMc1_?usp=sharing]

**Kaggle:**
[https://www.kaggle.com/code/adlerzaith/atividade08-dimmy-py]

**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
