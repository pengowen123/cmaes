set -o errexit -o nounset

if [ "$TRAVIS_BRANCH" != "master" ]
then
  echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
  exit 0
fi

rev=$(git rev-parse --short HEAD)

cd stage/_book

git init
git config user.name "Owen Sanchez"
git config user.email "pengowen816@gmail.com"

git remote add upstream "https://$GH_TOKEN@github.com/pengowen123/cmaes.git"
git fetch upstream
git reset upstream/gh-pages

touch .

git add -A .
git commit -m "rebuild pages at ${rev}"
git push -q upstream HEAD:gh-pages
