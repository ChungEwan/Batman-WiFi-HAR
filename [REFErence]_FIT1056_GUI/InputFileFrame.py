# Third party imports
import tkinter as tk
# from student_frame import StudentFrame
# from authenticator import authenticate
# from user import User
# from reset_pw import ResetPW
# from select_role import Role


# Local application imports


class LoginFrame(tk.Frame):
    """
    A LoginFrame class for all users of CodeVenture.
    """

    def __init__(self, master):
        """
        Constructor for the LoginFrame class.
        :param master: Tk object; the main window that the
                       login frame is to be contained.
        
        :syntax:
        In the context of graphical user interfaces (GUIs), "FocusIn" and "FocusOut" are events related to the focus of a user interface element, such as an input field or a button. Here's what each event means:

        FocusIn Event: This event occurs when a GUI element gains focus, which typically happens when a user interacts with it. For example, 
        when a user clicks on an input field or navigates to it using the keyboard, the input field gains focus, and the "FocusIn" event is triggered. 
        This event is often used to perform actions when an element becomes the active focus, 
        such as clearing placeholder text or changing the appearance of the focused element.

        FocusOut Event: This event occurs when a GUI element loses focus, which happens when the user clicks outside the element or navigates away from it. 
        For example, if a user clicks on an input field, enters some text, and then clicks somewhere else or presses the "Tab" key to move to the next input field, 
        the first input field loses focus, and the "FocusOut" event is triggered. This event is often used to perform actions when an element is no longer the active focus, 
        such as restoring placeholder text or validating user input.
        """
        # call the constructor from the parent class, which is tk.Frame
        super().__init__(master=master)
        # set the master attribute to the master parameter
        # which is our main interface / window
        self.master = master
        self.master.configure(bg="#ffffff")
        self.configure(bg="#ffffff")

        # Image obtained from:
        # https://www.veryicon.com/icons/healthcate-medical/medical-icon-two-color-icon/ico-health-clinic.html
        # set the image path to variable
        # image_path = "./images/changed.png"
        # # create a PhotoImage object from the image path, set the PhotoImage object to a class attribute
        # self.login_logo = tk.PhotoImage(file=image_path)


        tk.Label(master=self, 
                 #image=self.login_logo
                 bg="#ffffff", border=0).grid(row=0, column=0, rowspan=6,
                                                                                  columnspan=2, padx=10, pady=10)

        # Label containing the welcome heading
        login_title = tk.Label(master=self, fg="#347f8c",
                               text="Welcome to "
                                    "CodeVenture",
                               font=("Microsoft Yahei UI Light", 25, "bold"), bg="#ffffff")
        login_title.grid(row=0, column=2, columnspan=2, padx=10, pady=10)

        # Variable and entry for username
        self.username = tk.StringVar()
        self.username_entry = tk.Entry(border=0, master=self, textvariable=self.username,
                                       font=("Microsoft Yahei UI Light", 11), foreground="#3d95a5")
        # Position and layout the username entry field.
        self.username_entry.grid(row=1, column=2, columnspan=2, padx=50, pady=0, sticky="SEW")
        # Set the initial text in the entry field to "Username."
        self.username_entry.insert(0, "Username")
        # Bind the clear_username_entry function to the FocusIn event.
        self.username_entry.bind("<FocusIn>", self.clear_username_entry)
        # Bind the repopulate_username_entry function to the FocusOut event.
        self.username_entry.bind("<FocusOut>", self.repopulate_username_entry)

        # Create a frame as decoration underline for the username input
        username_underline = tk.Frame(master=self, bg="#347f8c", height=2)
        username_underline.grid(row=2, column=2, columnspan=2, sticky="NEW", padx=50, pady=0)
        # Lift (bring to the front) the username underline element
        username_underline.lift()



        # Variable and entry to password works the same way as username above
        self.password = tk.StringVar()
        self.password_entry = tk.Entry(border=0, master=self, textvariable=self.password,
                                       font=("Microsoft Yahei UI Light", 11), foreground="#3d95a5")
        self.password_entry.grid(row=3, column=2, columnspan=2, padx=50, pady=0, sticky="SEW")
        self.password_entry.insert(0, "Password")
        self.password_entry.bind("<FocusIn>", self.clear_password_entry)
        self.password_entry.bind("<FocusOut>", self.repopulate_password_entry)
        self.password_entry.bind("<KeyRelease>", self.musk_password_entry)

        # Create a frame as decoration underline for the password input
        password_underline = tk.Frame(master=self, bg="#347f8c", height=2)
        password_underline.grid(row=4, column=2, columnspan=2, sticky="NEW", padx=50, pady=0)
        # Lift (bring to the front) the username underline element
        password_underline.lift()


        # Button to login
        login_button = tk.Button(master=self, text="Login",
                                 command=self.authenticate_login, font=("Microsoft Yahei UI Light", 11), fg='#fff',
                                 bg="#347f8c", border=0)
        login_button.grid(row=5, column=2, columnspan=2, padx=50, pady=10, sticky=tk.EW)

        # Button to forgot password
        forgot_password_button = tk.Button(bg="#ffffff", cursor='hand2', master=self, text="Forgot Password",fg='#347f8c',
                                           command=self.forgot_password, relief="flat", borderwidth=0,
                                           font=("Microsoft Yahei UI Light", 10, "underline"))
        forgot_password_button.grid(row=6, column=2, padx=50, sticky=tk.W)

        # Button to sign up
        sign_up_button = tk.Button(bg="#ffffff", master=self, text="Sign Up", fg='#347f8c',relief="flat", cursor='hand2'
                                   , font=("Microsoft Yahei UI Light", 10, 'underline'), borderwidth=0, command=self.sign_up)
        sign_up_button.grid(row=6, column=3, padx=50, sticky=tk.E)

        # Variable and label to inform user of login outcome
        self.login_text = tk.StringVar()
        login_message = tk.Label(bg="#ffffff", fg='red', master=self, textvariable=self.login_text)

        login_message.grid(row=7, column=2, columnspan=2, padx=10, pady=10)

    def clear_username_entry(self, event):
        """
        Clears the username entry field when it receives focus.
        :param event: Event object
        :return: None
        """
        self.username_entry.delete(0, tk.END)

    def repopulate_username_entry(self, event=None):
        """
        Repopulates the username entry field with a default value if it's empty.
        :param event: Event object (optional)
        :return: None
        """
        if self.username_entry.get() == "":
            self.username_entry.insert(0, "Username")

    def clear_password_entry(self, event):
        """
        Clears the password entry field when it receives focus.
        :param event: Event object
        :return: None
        """
        self.password_entry.delete(0, tk.END)

    def repopulate_password_entry(self, event=None):
        """
        Repopulates the password entry field with a default value if it's empty.
        :param event: Event object (optional)
        :return: None
        """
        if self.password_entry.get() == "":
            self.password_entry.insert(0, "Password")



    def musk_password_entry(self, event):
        """
        Masks the password entry field when it's not set to the default value.
        :param event: Event object
        :return: None
        """
        if self.password_entry.get() != "Password":
            self.password_entry.config(show="*")

    def sign_up(self):
        """
        Allows the user to initiate the sign-up process. 
        This function prepares the user interface for registration by clearing any previous entries 
        in the username and password fields, and then navigates to the role selection page.
        
        :return: None
        """
        self.repopulate_password_entry()
        self.repopulate_username_entry()
        self.place_forget()
        # Create and display the Role selection page for the user to choose their role (e.g., student, teacher, parent).
        Role(self, self.master).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def forgot_password(self):
        """
        Allows the user to initiate the process of resetting their password. This function prepares the user interface for
        password reset by clearing any previous entries in the username and password fields. It then navigates to the
        password reset page where the user can reset their password.

        :return: None
        """
        self.repopulate_password_entry()
        self.repopulate_username_entry()
        self.place_forget()
        # Create and display the password reset page, where the user can reset their password.
        reset_frame = ResetPW(self, self.master)
        reset_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def authenticate_login(self):
        """
        Handles the login authentication process. This function is triggered when the user clicks the login button.
        It communicates with the authentication system to verify the provided username and password. If the user is
        successfully authenticated, it proceeds to grant access to the appropriate user role interface (e.g., StudentFrame).
        If authentication fails, it displays an error message to the user.

        :return: None
        # """
        # auth_res = authenticate(self.username.get(),
        #                         self.password.get())


        # if isinstance(auth_res, User):

        #     """Clear the username and password entry fields."""
        #     self.username_entry.delete(0, tk.END)  # Clear the username entry field
        #     self.password_entry.delete(0, tk.END)  # Clear the password entry field

        #     if auth_res.get_role() == "student":  # patient login
        #         self.login_text.set("")
        #         # Remove login page from display
        #         self.repopulate_password_entry()
        #         self.repopulate_username_entry()
        #         self.place_forget()

        #         # Create and display the Patient login frame
        #         # StudentFrame(self.master, self, auth_res).pack(fill=tk.BOTH, expand=True)

        #     elif auth_res.get_role() in ["parent", "teacher"]:
        #         self.login_text.set("other user's menu is not available yet")
        # else:
        #     self.login_text.set(auth_res)
        pass

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Login")
    root.geometry("900x540")
    root.resizable(width=False, height=False)
    root.configure(bg="#ffffff")
    LoginFrame(root).place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    root.mainloop()
