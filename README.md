# pair-class-verification on Google Colab
# Compile with OpenMP support for parallel processing
!g++ -std=c++17 -O3 -fopenmp verify_relations.cpp -o verify_relations

# Run with different maximum values
!./verify_relations 100000  # 10^5 for quick test
!./verify_relations 1000000  # 10^6 for medium test
!./verify_relations 10000000  # 10^7 for full test
