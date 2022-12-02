# TAREA-3
Resolución de ecuaciones diferenciales

Comenzamos Resolviendo el inciso a) (la función trigonométrica). El código no compila debido a 'method' object is not iterable.

El inciso a, consta de límites de -1 a 1. Implementamos 5 capas a la red neuronal, con 500 epocas y activación tanh y linear.

Se logró compilar los códigos teniendo los resultados siguientes:

Para el inciso a) del problema 2 obtuvimos una aproximación más que aceptable, con un loss final de 0.0241.

Para el inciso b) del problema 2 obtuvimos una aproximación aún mejor, con un valor de loss final de 0.0016, prácticamente despreciable.

Para el inciso a) del problema 1 obtuvimos una aproximación muy buena, un buen entrenamiento, con un valor de loss final de 0.016.

Obtuvimos un inconveniente para el problema 1 inciso b), ya que persistió el problema 'method' object is not iterable y por ende no pudimos compilar, pero corrigiendo ese detalle sutil se confía en tener una red con un valor similar de loss final como en los otros 3 codigos realizados.