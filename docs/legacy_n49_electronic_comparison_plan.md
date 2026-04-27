# Reinicio de comparacion `TDNEGF_exciton_B` vs `TDNEGF`

Quiero reiniciar desde cero la comparacion entre `TDNEGF_exciton_B` (codigo viejo) y `TDNEGF` (codigo nuevo).

No asumas que las conclusiones previas del chat son correctas. Tomalas solo como hipotesis a verificar.

## Objetivo

Hacer una validacion inicial minima, puramente electronica, en la que `TDNEGF` reproduzca lo mas exactamente posible el mismo problema electronico de `TDNEGF_exciton_B`, antes de considerar LLG o dinamica de espines clasicos.

Para esta etapa:

- no incluyas LLG
- no incluyas acoplo espin-clasico
- enfocate solo en geometria, base, leads/contactos, Hamiltoniano electronico, embedding/self-energy y observables electronicos
- la reproduccion en `TDNEGF` debe hacerse con polos `N49`

## Tareas requeridas

1. Reconstruye primero cual era el problema fisico original en `TDNEGF_exciton_B`.
   No asumas que los scripts actuales ya son equivalentes.

2. Crea un script nuevo en `TDNEGF` que reproduzca explicitamente ese mismo caso electronico legacy.
   No alcanza con reutilizar scripts existentes si no fijan de manera inequivoca la misma geometria, base y contactos.

3. Antes de correr nada, verifica de forma explicita que el script nuevo reproduce exactamente:
   - `Nx`, `Ny`
   - numero total de sitios
   - `N_orb`
   - numero de componentes/espines `Nσ`
   - numero de leads/canales `Nα`
   - orden de la base
   - conectividad exacta entre sitios
   - correspondencia sitio/orbital/espin entre codigo viejo y codigo nuevo
   - correspondencia exacta de leads/contactos entre ambos codigos

4. Haz un mapa explicito de parametros y convenciones antes de correr.
   Para cada parametro relevante, indica:
   - nombre en `TDNEGF_exciton_B`
   - nombre en `TDNEGF`
   - significado fisico
   - unidades
   - valor usado
   - si requiere conversion
   - si el mapeo es seguro o incierto

5. Revisa explicitamente estos puntos:
   - geometria y base: `Nx`, `Ny`, sitios totales, `N_orb`, `Nσ`, `Nα`, orden de base
   - Hamiltoniano: hoppings, onsites, separacion de espin, definicion de `H(t)`
   - embedding/self-energy/polos: `N_λ1`, `N_λ2`, polos, pesos/residuos, `χ`, `Σ`, `Γ`
   - diferencia entre embedding legacy exacto y embedding reconstruido
   - dinamica temporal electronica: `dt`, tiempo total, integrador, condiciones iniciales, frecuencia de muestreo
   - observables electronicos: corriente, densidad, traza y cualquier observable que se compare entre ambos codigos

6. La validacion debe hacerse por etapas:
   - primero: benchmark electronico minimo, con la geometria exacta y LLG apagado
   - despues: comparacion de embedding/self-energy
   - solo despues, si todo cierra, recien considerar el caso completo con LLG

7. En cada etapa define:
   - inputs congelados
   - observables a comparar
   - criterio de exito/fallo
   - posibles fuentes de discrepancia

8. Si aparecen diferencias, no concluyas enseguida que un codigo esta mal.
   Primero revisa:
   - mapeo geometrico exacto
   - mapeo exacto de leads/contactos
   - unidades y prefactores
   - signos, conjugaciones y orden de indices
   - definicion de corriente, densidad, traza y magnetizacion
   - diferencia entre embedding legacy exacto y embedding reconstruido

## Consejos importantes ya aprendidos aqui

- No des por equivalente una "geometria generica" del codigo nuevo con la geometria legacy solo porque `Nx` y `Ny` parecen coincidir.
  Tienes que verificar tambien:
  - el numero real de grados de libertad
  - el orden exacto de la base
  - que significa cada indice en ambos codigos
  - como se empaquetan sitio, orbital y espin

- No asumas que los leads del codigo nuevo se colocan automaticamente de forma equivalente al legacy.
  En este problema, la correspondencia de contactos tiene que revisarse explicitamente indice por indice.

- Presta mucha atencion a si el codigo viejo usa un patron hardcodeado de contactos/leads que no coincide automaticamente con el constructor "natural" del codigo nuevo.

- Agrega tambien consejos explicitos sobre la geometria del sistema y sobre el lead/contacto segun lo que ya aprendimos en este proyecto, para evitar falsos acuerdos por una geometria mal interpretada o por contactos mal mapeados.

## Entregable final esperado

Al final entrega un Jupyter notebook donde se muestre que:

- la corriente da igual entre `TDNEGF_exciton_B` y `TDNEGF`
- la densidad en el sitio da cerca

Ese notebook debe dejar clara la comparacion del benchmark electronico inicial sin LLG, usando la reproduccion con polos `N49`.
