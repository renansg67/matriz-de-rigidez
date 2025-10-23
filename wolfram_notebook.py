mathematica_code = """
(* Definição de constantes e dimensões em mm *)
Lmenor = {L_MENOR_FIXO};
Htronco = {H_TRONCO_FIXO};
Hprisma = {H_PRISMA_FIXO};

(* Lmaior é forçado para ser a altura do prisma, para que as faces do prisma sejam quadradas *)
Lmaior = Hprisma;

(* Raio do Octógono (Circunraio R = L / (2 Sin[Pi/8])) *)
R[L_] := L / (2 Sin[Pi/8]);
Rmenor = R[Lmenor]; (* {poly_dims['R_menor']:.4f} mm *)
Rmaior = R[Lmaior]; (* {poly_dims['R_maior']:.4f} mm *)

(* Alturas Z dos 4 anéis de 8 vértices *)
Z1 = 0;
Z2 = Htronco; (* {H_TRONCO_FIXO:.1f} *)
Z3 = Htronco + Hprisma; (* {H_TRONCO_FIXO + H_PRISMA_FIXO:.1f} *)
Z4 = 2 Htronco + Hprisma; (* {poly_dims['Z4']:.1f} *)

(* Geração de coordenadas para um octógono de raio R no plano XY *)
Vcoord[R_, Z_] := Table[
    {{N[R Cos[phi]], N[R Sin[phi]], Z}},
    {{phi, N[Pi/8], N[2 Pi - Pi/8], N[Pi/4]}}
];

(* As 32 Coordenadas dos Vértices *)
vertices = Join[
    Vcoord[Rmenor, Z1],  (* V1 a V8: Base Menor, Z=0 *)
    Vcoord[Rmaior, Z2],  (* V9 a V16: Seção Maior Inferior, Z={H_TRONCO_FIXO:.1f} *)
    Vcoord[Rmaior, Z3],  (* V17 a V24: Seção Maior Superior, Z={H_TRONCO_FIXO + H_PRISMA_FIXO:.1f} *)
    Vcoord[Rmenor, Z4]   (* V25 a V32: Topo Menor, Z={poly_dims['Z4']:.1f} *)
];

(* Lista de Faces (2 Octogonais + 24 Quadrilaterais) *)
faces = Join[
    (* F1: Base Octogonal Menor (V1-V8) *)
    {{Range[8]}},
    (* F2: Topo Octogonal Menor (V25-V32) *)
    {{Range[25, 32]}},
    
    (* F3-F10: Tronco Inferior (V1-V8 para V9-V16) *)
    Table[{{i, i + 8, If[i < 8, i + 9, 9], If[i < 8, i + 1, 1]}}, {{i, 1, 8}}],
    (* F11-F18: Prisma Central (V9-V16 para V17-V24) - Retângulos *)
    Table[{{i + 8, i + 16, If[i < 8, i + 17, 17], If[i < 8, i + 9, 9]}}, {{i, 1, 8}}],
    (* F19-F26: Tronco Superior (V17-V24 para V25-V32) *)
    Table[{{i + 16, i + 24, If[i < 8, i + 25, 25], If[i < 8, i + 17, 17]}}, {{i, 1, 8}}]
];

(* Comando de Renderização do Poliedro no Mathematica *)
Polyhedron[vertices, faces,
    PlotLabel -> "Poliedro de 26 Faces (Tronco-Prisma-Tronco)",
    Boxed -> True,
    FaceGrids -> All
]
"""