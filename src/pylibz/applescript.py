
def applescript():
	"""
	
	https://sspai.com/post/46912

	https://sspai.com/post/43758
	"""
	return """

```applescript
**use** AppleScript version "2.4" -- Yosemite (10.10) or later

**use** _scripting additions_

  

-- 告诉 System Events 我们要和它交互

**tell** _application_ "System Events"

	**tell** _process_ "网易有道翻译"
		
		-- set frontmost to true
		
			**tell** _window_ "网易有道翻译"
			
				**tell** _scroll area_ 1 **of** _group_ 1 **of** _group_ 1 -- 滚动区 组 组
					
					**tell** _UI element_ 1 -- 组 HTML 内容
						
						entire contents -- 获取所有 UI 元素
						
						-- static text "hierarchy"
					
					**set** value **of** _group_ 7 **to** "aaa"
				
				**end** **tell**
			
			**end** **tell**
		
		**end** **tell**
	
	**end** **tell**

**end** **tell**



  

-- 告诉 System Events 我们要和它交互

tell application "System Events"

	tell process "网易有道翻译"
	
		entire contents -- 获取所有UI
	
	end tell

end tell

-- static text "hierarchy" of group 7 of UI element 1 of scroll area 1 of group 1 of group 1 of window "网易有道翻译" of application process "网易有道翻译" of application "System Events",
```

"""
