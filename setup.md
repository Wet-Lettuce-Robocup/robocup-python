To back up sd card:
Find disk name with `diskutil list`
Run `sudo dd if=/dev/rdisk4 of=/Volumes/<folder>/pi_backup.img bs=1m status=progress` 
Pishrink https://github.com/Drewsif/pishrink

On Pi:
```bash
# Initial setup and upgrading packages
sudo apt update && sudo apt full-upgrade -y
sudo apt autoremove -y
sudo apt clean
sudo reboot

# Install neovim + lazyvim + tmux
sudo apt install -y git cmake build-essential gettext curl ripgrep fd-find fzf tmux
git clone https://github.com/neovim/neovim.git
cd neovim
git checkout stable
make CMAKE_BUILD_TYPE=RelWithDebInfo
cd build && cpack -G DEB
sudo dpkg -i nvim-linux-arm64.deb
cd ~
sudo rm -rf neovim

git clone https://github.com/LazyVim/starter ~/.config/nvim
nvim

# Install zsh and oh-my-zsh
sudo apt install zsh -y
zsh
```
```zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k"
nvim .zshrc # Change ZSH_THEME to ZSH_THEME="powerlevel10k/powerlevel10k"
source ~/.zshrc # Go through the customisation of powerlevel10k
```

Now some nice addons for oh-my-zsh:
```zsh
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

and add `zsh-autosuggestions` and `zsh-syntax-highlighting` to 'plugins' within .zshrc

Make sure that raspi-config is set up correctly, including enabling i2c and spi

Add this to boot/firmware/config.txt:
```bash
#fan config
dtparam=fan_temp0=45000,fan_temp0_hyst=2000,fan_temp0_speed=50
dtparam=fan_temp1=60000,fan_temp1_hyst=3000,fan_temp1_speed=80
dtparam=fan_temp2=70000,fan_temp2_hyst=4000,fan_temp1_speed=120
dtparam=fan_temp3=80000,fan_temp3_hyst=5000,fan_temp1_speed=255
```

Clone the repository:
```zsh
git clone https://github.com/Wet-Lettuce-Robocup/robocup-python.git
```
(Make sure that the right branch is checked out)

Set up custom PWM device tree:
```zsh
cd device_tree
./install.sh
cd ..
```

Create venv:
```zsh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python adafruit-circuitpython-vl53l1x luma.oled

```