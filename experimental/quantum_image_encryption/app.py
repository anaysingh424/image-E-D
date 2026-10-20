import hashlib
import io
import numpy as np
import streamlit as st
from PIL import Image
from quantum_core import quantum_keystream, encrypt_array, decrypt_array, metrics

st.set_page_config(page_title="Quantum Image Encryption And Decryption", page_icon="🔐", layout="wide")
st.title("🔐 Quantum Image Encryption and Decryption")
st.caption("Educational hybrid prototype: quantum-circuit-derived keystream + reversible pixel transformation")

st.warning("Research/demo only — not production-grade encryption. The image transformations are classical, and the simulated circuit is not a security guarantee. Use AES-GCM for real confidentiality.")

with st.sidebar:
    st.header("Configuration")
    password = st.text_input("Secret Passphrase", type="password", help="Use the same passphrase to decrypt.")
    image_mode = st.selectbox("Processing Mode", ["Grayscale (Recommended)", "RGB"])
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.write("Image → quantum circuit simulation → deterministic keystream → pixel permutation/XOR → encrypted image")

uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    original = Image.open(uploaded).convert("L" if image_mode.startswith("Grayscale") else "RGB")
    arr = np.array(original, dtype=np.uint8)
    left, right = st.columns(2)
    with left:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)
        st.caption(f"Dimensions: {original.width} × {original.height} · Channels: {1 if arr.ndim == 2 else 3}")
    if not password:
        st.info("Enter a secret passphrase in the sidebar to run the demo.")
    else:
        if st.button("Encrypt Image", type="primary"):
            cipher, meta = encrypt_array(arr, password)
            st.session_state["cipher"] = cipher
            st.session_state["meta"] = meta
            st.session_state["shape"] = arr.shape
            st.session_state["mode"] = image_mode
            st.session_state["original_hash"] = hashlib.sha256(arr.tobytes()).hexdigest()
        if "cipher" in st.session_state and st.session_state.get("shape") == arr.shape and st.session_state.get("mode") == image_mode:
            cipher = st.session_state["cipher"]
            with right:
                st.subheader("Encrypted image (visualization)")
                view = Image.fromarray(cipher)
                st.image(view, use_container_width=True)
                st.caption("Cipher pixels are shown directly; a PNG preview is not a cryptographic file format.")
            buf = io.BytesIO()
            Image.fromarray(cipher).save(buf, format="PNG")
            st.download_button("Download Encrypted Pixel Image", buf.getvalue(), "encrypted_demo.png", "image/png")
            st.markdown("---")
            st.subheader("Decrypt and Verify")
            if st.button("Decrypt with Entered Passphrase"):
                recovered = decrypt_array(cipher, password, st.session_state["meta"])
                recovered_img = Image.fromarray(recovered)
                st.image(recovered_img, caption="Recovered image", use_container_width=True)
                ok = np.array_equal(recovered, arr)
                st.success("Exact recovery verified." if ok else "Recovery did not match. Check the passphrase and image settings.")
                st.write("Recovered-image SHA-256:", hashlib.sha256(recovered.tobytes()).hexdigest())
                st.write("Original-image SHA-256:", st.session_state["original_hash"])
        st.caption("If you change the passphrase after encrypting, decryption should fail to reproduce the original.")
else:
    st.info("Upload a sample image to begin. For a fast first run, use a small image (e.g. 128 × 128).")

with st.expander("What is quantum about this prototype?"):
    st.markdown("""
    - A small parameterized quantum circuit is simulated to produce a deterministic measurement-derived bitstream.
    - The bitstream seeds a reproducible stream used by the demo's classical reversible pixel operations.
    - Image bytes are **not** stored as qubits, and the app does **not** claim quantum advantage or proven quantum security.
    - For actual secure storage or transfer, use a vetted authenticated cipher such as AES-GCM and proper key management.
    """)
