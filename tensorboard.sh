#!/bin/bash

ssh bluecrystal -L 6006:$1

URL="http://localhost:6006"

if which xdg-open > /dev/null
then
  xdg-open $URL
elif which gnome-open > /dev/null
then
  gnome-open $URL
fi
