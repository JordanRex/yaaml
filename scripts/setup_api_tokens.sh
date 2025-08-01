#!/bin/bash

# 🔑 YAAML API Token Setup Helper
# This script helps you set up PyPI API tokens for automated publishing

echo "🚀 YAAML API Token Setup Helper"
echo "================================"
echo ""

# Check if user wants to set up Test PyPI
echo "🧪 Test PyPI Setup (Recommended first step)"
echo "Would you like to set up Test PyPI publishing? (y/n)"
read -r setup_test_pypi

if [[ $setup_test_pypi == "y" || $setup_test_pypi == "Y" ]]; then
    echo ""
    echo "📋 Test PyPI Setup Instructions:"
    echo "1. Go to: https://test.pypi.org/manage/account/"
    echo "2. Scroll down to 'API tokens'"
    echo "3. Click 'Add API token'"
    echo "4. Token name: 'GitHub Actions - YAAML'"
    echo "5. Scope: 'Entire account (not recommended for production)'"
    echo "6. Click 'Add token'"
    echo "7. Copy the token (starts with pypi-)"
    echo ""
    echo "🔧 GitHub Secret Setup:"
    echo "1. Go to your GitHub repo: https://github.com/JordanRex/yaaml"
    echo "2. Settings → Secrets and variables → Actions"
    echo "3. Click 'New repository secret'"
    echo "4. Name: TEST_PYPI_API_TOKEN"
    echo "5. Secret: [paste your token]"
    echo "6. Click 'Add secret'"
    echo ""
fi

# Check if user wants to set up Production PyPI
echo "🚀 Production PyPI Setup (For v1.0.0+ releases)"
echo "Would you like to set up Production PyPI publishing? (y/n)"
read -r setup_prod_pypi

if [[ $setup_prod_pypi == "y" || $setup_prod_pypi == "Y" ]]; then
    echo ""
    echo "📋 Production PyPI Setup Instructions:"
    echo "1. Go to: https://pypi.org/manage/account/"
    echo "2. Scroll down to 'API tokens'"
    echo "3. Click 'Add API token'"
    echo "4. Token name: 'GitHub Actions - YAAML'"
    echo "5. Scope: 'Entire account (not recommended for production)'"
    echo "   Note: You can create project-specific tokens later"
    echo "6. Click 'Add token'"
    echo "7. Copy the token (starts with pypi-)"
    echo ""
    echo "🔧 GitHub Secret Setup:"
    echo "1. Go to your GitHub repo: https://github.com/JordanRex/yaaml"
    echo "2. Settings → Secrets and variables → Actions"
    echo "3. Click 'New repository secret'"
    echo "4. Name: PYPI_API_TOKEN"
    echo "5. Secret: [paste your token]"
    echo "6. Click 'Add secret'"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Next Steps:"
echo "- For v0.x.x releases: Use Test PyPI only (safe for testing)"
echo "- For v1.x.x releases: Use Production PyPI (public packages)"
echo "- The workflow will automatically detect which tokens are available"
echo ""
echo "💡 Pro tip: Set up Test PyPI first and test with v0.1.0 before going to production!"
