import sys
import traceback

try:
    # Import and run the main function
    sys.path.insert(0, 'src')
    from main import main
    print("Starting exam cheating detection system...")
    main()
except Exception as e:
    print("\n" + "="*60)
    print("ERROR OCCURRED:")
    print("="*60)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    print("="*60)
    traceback.print_exc()
    print("="*60)
    input("\nPress Enter to exit...")
    sys.exit(1)
