with open("turboquant/runtime/attention.py", "r") as f:
    text = f.read()

text = text.replace("main_scores = q_rot @ mx.swapaxes(main_rot, -1, -2)", """
    if q_rot.shape[-3] != main_rot.shape[-3]:
        n_rep = q_rot.shape[-3] // main_rot.shape[-3]
        main_rot = mx.repeat(main_rot, n_rep, axis=-3)

    main_scores = q_rot @ mx.swapaxes(main_rot, -1, -2)
""")

text = text.replace("return main_scores + resid_scores", """
        if q_rot.shape[-3] != resid_rot.shape[-3]:
            resid_rot = mx.repeat(resid_rot, q_rot.shape[-3] // resid_rot.shape[-3], axis=-3)
        resid_scores = q_rot @ mx.swapaxes(resid_rot, -1, -2)
        return main_scores + resid_scores
""", 1) # Only replace the topk one

with open("turboquant/runtime/attention.py", "w") as f:
    f.write(text)
