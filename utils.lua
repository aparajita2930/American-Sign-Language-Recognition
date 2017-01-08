--[[
File contains different helper methods
]]--

local utils = {}

function utils.getArgs(args, name, default)
  if args == nil then args = {} end
  if args[name] == nil and default == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif args[name] == nil then
    return default
  else
    return args[name]
  end
end

-- Helper method to print message, along with time
function utils.print(message)
  local _time = os.date("*t")
  local _currTime = ("%02d:%02d:%02d"):format(_time.hour, _time.min, _time.sec)
  print(string.format("%s %s",_currTime, message))
end

return utils
