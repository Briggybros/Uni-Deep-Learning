#!/bin/bash

URL="http://localhost:6006"

if which xdg-open > /dev/null
then
  xdg-open $URL > /dev/null 2> /dev/null &
elif which gnome-open > /dev/null
then
  gnome-open $URL > /dev/null 2> /dev/null &
fi

ssh bluecrystal -L 6006:$1
