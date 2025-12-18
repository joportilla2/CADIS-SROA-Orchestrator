from cadis_sroa_rep2.argumentation import AF, Argument, grounded_extension, induced_defeat_graph

def test_grounded_simple_attack():
    af = AF()
    af.add_argument(Argument("a","A1","H1",0.8))
    af.add_argument(Argument("b","A2","H2",0.7))
    af.add_attack("b","a")
    G = grounded_extension(af)
    assert "b" in G
    assert "a" not in G

def test_hybrid_filters_weak_attack():
    af = AF()
    af.add_argument(Argument("a","A1","H1",0.8))
    af.add_argument(Argument("b","A2","H2",0.9))
    af.add_attack("b","a")
    weights = {"a": 1.0, "b": 0.4}
    rep = induced_defeat_graph(af, weights, lam=1.25)
    assert len(rep.attacks) == 0
    G = grounded_extension(rep)
    assert "a" in G and "b" in G
