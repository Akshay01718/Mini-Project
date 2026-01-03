try:
    import mediapipe.python.solutions.face_mesh
    print("Explicit_Import_SUCCESS")
except ImportError:
    try:
        import mediapipe.solutions.face_mesh
        print("Standard_Import_SUCCESS")
    except Exception as e:
        print(f"Standard_Import_FAILED: {e}")
except Exception as e:
    print(f"Explicit_Import_FAILED: {e}")
